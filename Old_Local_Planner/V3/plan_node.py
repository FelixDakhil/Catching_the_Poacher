#!/usr/bin/env python3
"""
plan_node.py  –  Local motion planner for TurtleBot3 in Gazebo.

Algorithm: Vector Field Histogram+  (VFH+)
-------------------------------------------
VFH+ is a reactive local planner that:
  1. Builds a polar obstacle density histogram from the laser scan.
  2. Smoothes the histogram with a sliding window.
  3. Identifies "valleys" (angular sectors free of obstacles).
  4. Selects the candidate direction closest to the goal heading.
  5. Outputs a (linear_speed, angular_speed) command.

Subscriptions
-------------
  /processed_scan   Float32MultiArray   – cleaned ranges from see_node
  /scan_metadata    Float64MultiArray   – angle_min, angle_max, angle_incr, range_max
  /goal_heading     Float64             – desired heading in radians (world frame)
                                          publish -pi … pi; 0 = straight ahead.
                                          If nobody publishes, the robot drives
                                          forward with obstacle avoidance only.

Publication
-----------
  /vfh_command      Float64MultiArray   – [linear_vel, angular_vel]
                                          consumed by act_node.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, Float64MultiArray, Float64


# ---------------------------------------------------------------------------
# VFH+ tuning defaults (all exposed as ROS parameters)
# ---------------------------------------------------------------------------
VFH_ALPHA          = 5       # degrees per histogram sector
VFH_THRESHOLD      = 0.20    # obstacle density threshold (0–1 scale)
VFH_SMOOTH_WIN     = 3       # sectors on each side used for smoothing
VFH_D_MAX          = 3.5     # max sensor range (m), overridden by metadata
VFH_ROBOT_RADIUS   = 0.18    # TurtleBot3 Burger radius (m)
VFH_SAFETY_DIST    = 0.25    # clearance added on top of robot radius (m)
VFH_A              = 1.0     # obstacle weight coefficient  (h = a - b*d)
VFH_B              = 1.0 / VFH_D_MAX
MAX_LINEAR_VEL     = 0.22    # m/s  (TurtleBot3 Burger hardware limit)
MAX_ANGULAR_VEL    = 2.84    # rad/s
LINEAR_SCALE       = 0.8     # fraction of max used when path is clear


class VFHPlanner:
    """
    Pure-Python VFH+ implementation (no ROS dependencies).

    Parameters
    ----------
    alpha_deg       : sector width in degrees
    threshold       : POD value above which a sector is "blocked"
    smooth_window   : number of neighbouring sectors used for smoothing
    d_max           : maximum sensor range (metres)
    robot_radius    : physical radius of the robot (metres)
    safety_dist     : extra clearance added to robot radius (metres)
    a, b            : obstacle magnitude coefficients
    """

    def __init__(
        self,
        alpha_deg:    float = VFH_ALPHA,
        threshold:    float = VFH_THRESHOLD,
        smooth_window: int  = VFH_SMOOTH_WIN,
        d_max:        float = VFH_D_MAX,
        robot_radius: float = VFH_ROBOT_RADIUS,
        safety_dist:  float = VFH_SAFETY_DIST,
        a:            float = VFH_A,
        b:            float = VFH_B,
    ) -> None:
        self.alpha     = math.radians(alpha_deg)
        self.threshold = threshold
        self.smooth_w  = smooth_window
        self.d_max     = d_max
        self.a         = a
        self.b         = b
        # Enlarged robot radius used when masking sectors
        self.r_enlarged = robot_radius + safety_dist
        # Number of sectors covering 360°
        self.num_sectors = max(1, round(2 * math.pi / self.alpha))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        ranges:          list[float],
        angle_min:       float,
        angle_increment: float,
        goal_heading:    float,
    ) -> tuple[float, float]:
        """
        Run one VFH+ cycle.

        Parameters
        ----------
        ranges          : cleaned range readings (metres)
        angle_min       : angle of first beam (radians, robot frame)
        angle_increment : angular step between beams (radians)
        goal_heading    : desired travel direction (radians, robot frame)

        Returns
        -------
        (linear_vel, angular_vel) in m/s and rad/s
        """
        # 1. Build raw polar obstacle density histogram
        hist = self._build_histogram(ranges, angle_min, angle_increment)

        # 2. Smooth
        hist_smooth = self._smooth(hist)

        # 3. Binary: blocked / free
        binary = [1 if h > self.threshold else 0 for h in hist_smooth]

        # 4. Find candidate valleys
        valleys = self._find_valleys(binary)

        # 5. Select best direction
        best_dir = self._select_direction(valleys, goal_heading)

        # 6. Convert to velocity commands
        return self._to_velocities(best_dir, hist_smooth, goal_heading)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_histogram(
        self,
        ranges:          list[float],
        angle_min:       float,
        angle_increment: float,
    ) -> list[float]:
        """Map each beam to a histogram sector and accumulate obstacle magnitude."""
        hist = [0.0] * self.num_sectors

        for i, d in enumerate(ranges):
            if d >= self.d_max:
                continue  # free space – skip

            # Beam angle in robot frame, wrapped to [-pi, pi]
            beam_angle = angle_min + i * angle_increment
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))

            # Sector index  (0 = front, increasing counter-clockwise)
            sector = int((beam_angle + math.pi) / self.alpha) % self.num_sectors

            # Obstacle magnitude: closer = higher weight
            magnitude = (self.a - self.b * d) * (self.a - self.b * d)
            hist[sector] = max(hist[sector], magnitude)

        # Normalise to [0, 1]
        max_h = max(hist) if max(hist) > 0.0 else 1.0
        return [h / max_h for h in hist]

    def _smooth(self, hist: list[float]) -> list[float]:
        """Symmetric sliding-window average (circular)."""
        n   = len(hist)
        out = [0.0] * n
        l   = self.smooth_w
        for k in range(n):
            total = 0.0
            count = 0
            for j in range(-l, l + 1):
                total += hist[(k + j) % n]
                count += 1
            out[k] = total / count
        return out

    def _find_valleys(self, binary: list[int]) -> list[tuple[int, int]]:
        """
        Return list of (start_sector, end_sector) tuples for free valleys.
        Sectors are 0-indexed; ranges are inclusive.
        """
        n       = len(binary)
        valleys = []
        start   = None

        # Duplicate the array to handle wrap-around
        extended = binary + binary
        for i in range(2 * n):
            if extended[i] == 0 and start is None:
                start = i
            elif extended[i] == 1 and start is not None:
                # Record valley only if it fits in first n sectors
                if start < n:
                    valleys.append((start % n, (i - 1) % n))
                start = None

        # Close a valley that runs to the end
        if start is not None and start < n:
            valleys.append((start % n, (2 * n - 1) % n))

        return valleys

    def _sector_to_angle(self, sector: int) -> float:
        """Convert a sector index to its centre angle (radians, robot frame)."""
        angle = sector * self.alpha - math.pi + self.alpha / 2.0
        return math.atan2(math.sin(angle), math.cos(angle))

    def _select_direction(
        self,
        valleys:      list[tuple[int, int]],
        goal_heading: float,
    ) -> float | None:
        """
        Pick the steering direction (radians) that is:
          - inside a free valley
          - closest to goal_heading
        Returns None if no valley found (full blockage).
        """
        if not valleys:
            return None

        # Goal sector
        goal_sector = int((goal_heading + math.pi) / self.alpha) % self.num_sectors

        best_dir  = None
        best_cost = float("inf")

        for v_start, v_end in valleys:
            # Centre of valley
            if v_end >= v_start:
                valley_sectors = list(range(v_start, v_end + 1))
            else:
                n = self.num_sectors
                valley_sectors = list(range(v_start, n)) + list(range(0, v_end + 1))

            # Candidate: closest valley sector to goal
            for s in valley_sectors:
                diff = abs(s - goal_sector)
                diff = min(diff, self.num_sectors - diff)  # wrap
                if diff < best_cost:
                    best_cost = diff
                    best_dir  = self._sector_to_angle(s)

        return best_dir

    def _to_velocities(
        self,
        direction:    float | None,
        hist_smooth:  list[float],
        goal_heading: float,
    ) -> tuple[float, float]:
        """Convert chosen direction to (v, w) commands."""
        # Emergency stop: no valley found
        if direction is None:
            return 0.0, MAX_ANGULAR_VEL * 0.5  # spin to find opening

        # Angular velocity: proportional controller
        err = direction
        err = math.atan2(math.sin(err), math.cos(err))  # wrap

        # Clamp angular velocity
        angular_vel = float(np.clip(
            2.0 * err,   # Kp = 2.0
            -MAX_ANGULAR_VEL,
            MAX_ANGULAR_VEL,
        ))

        # Linear velocity: reduce when turning sharply or near obstacles
        turn_factor  = 1.0 - min(1.0, abs(err) / math.pi)
        front_sector = self.num_sectors // 2   # sector facing forward
        # Average density of the 3 front sectors
        front_density = sum(
            hist_smooth[(front_sector + d) % self.num_sectors]
            for d in (-1, 0, 1)
        ) / 3.0
        obstacle_factor = 1.0 - min(1.0, front_density)

        linear_vel = float(np.clip(
            MAX_LINEAR_VEL * LINEAR_SCALE * turn_factor * obstacle_factor,
            0.0,
            MAX_LINEAR_VEL,
        ))

        return linear_vel, angular_vel


# ---------------------------------------------------------------------------
# ROS 2 node wrapper
# ---------------------------------------------------------------------------
class PlanNode(Node):
    """ROS 2 wrapper around VFHPlanner."""

    def __init__(self) -> None:
        super().__init__("plan_node")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("alpha_deg",    VFH_ALPHA)
        self.declare_parameter("threshold",    VFH_THRESHOLD)
        self.declare_parameter("smooth_window", VFH_SMOOTH_WIN)
        self.declare_parameter("d_max",        VFH_D_MAX)
        self.declare_parameter("robot_radius", VFH_ROBOT_RADIUS)
        self.declare_parameter("safety_dist",  VFH_SAFETY_DIST)
        self.declare_parameter("vfh_a",        VFH_A)
        self.declare_parameter("vfh_b",        VFH_B)

        self._planner = VFHPlanner(
            alpha_deg    = self.get_parameter("alpha_deg").value,
            threshold    = self.get_parameter("threshold").value,
            smooth_window= self.get_parameter("smooth_window").value,
            d_max        = self.get_parameter("d_max").value,
            robot_radius = self.get_parameter("robot_radius").value,
            safety_dist  = self.get_parameter("safety_dist").value,
            a            = self.get_parameter("vfh_a").value,
            b            = self.get_parameter("vfh_b").value,
        )

        # ---- state --------------------------------------------------------
        self._ranges:    list[float] | None = None
        self._angle_min: float = -math.pi
        self._angle_inc: float =  math.radians(1.0)
        self._range_max: float =  VFH_D_MAX
        self._goal_heading: float = 0.0   # drive straight by default

        # ---- subscribers --------------------------------------------------
        self._scan_sub = self.create_subscription(
            Float32MultiArray,
            "/processed_scan",
            self._scan_cb,
            10,
        )
        self._meta_sub = self.create_subscription(
            Float64MultiArray,
            "/scan_metadata",
            self._meta_cb,
            10,
        )
        self._goal_sub = self.create_subscription(
            Float64,
            "/goal_heading",
            self._goal_cb,
            10,
        )

        # ---- publisher ----------------------------------------------------
        self._cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/vfh_command",
            10,
        )

        self.get_logger().info("PlanNode ready  |  VFH+ local planner active")

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------
    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._angle_min = msg.data[0]
            # angle_max      = msg.data[1]  (not stored; derived from inc × N)
            self._angle_inc = msg.data[2]
            self._range_max = msg.data[3]
            self._planner.d_max = msg.data[3]

    def _goal_cb(self, msg: Float64) -> None:
        self._goal_heading = float(msg.data)

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        """Main planning trigger – called every time a new scan arrives."""
        self._ranges = list(msg.data)

        if not self._ranges:
            return

        lin, ang = self._planner.compute(
            ranges          = self._ranges,
            angle_min       = self._angle_min,
            angle_increment = self._angle_inc,
            goal_heading    = self._goal_heading,
        )

        out = Float64MultiArray()
        out.data = [lin, ang]
        self._cmd_pub.publish(out)

        self.get_logger().debug(
            f"VFH+ command  →  v={lin:.3f} m/s   ω={ang:.3f} rad/s"
        )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
