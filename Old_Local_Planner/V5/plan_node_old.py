#!/usr/bin/env python3
"""
plan_node.py  –  Local motion planner for TurtleBot3 in Gazebo.

Algorithm: Vector Field Histogram+  (VFH+)
-------------------------------------------
VFH+ reactive local planner:
  1. Build a polar obstacle density histogram from the laser scan.
  2. Smooth the histogram with a sliding window.
  3. Identify "valleys" (angular sectors free of obstacles).
  4. Select the candidate direction closest to the goal heading.
  5. Output a (linear_speed, angular_speed) command.

NEW in this version
-------------------
* Goal input is now a geometry_msgs/Point on /goal_point  (x, y in odom frame).
  The node subscribes to /odom to obtain the robot's current pose and derives
  the local goal heading automatically.
* Linear velocity is FIXED (fixed-wing style) – the robot always moves forward
  at a constant speed; only the heading changes.
* Arrival detection: when the robot is within `arrival_radius` of the goal,
  a zero-velocity command is published and the goal is cleared.

Subscriptions
-------------
  /processed_scan   Float32MultiArray   – cleaned ranges from see_node
  /scan_metadata    Float64MultiArray   – [angle_min, angle_max, angle_incr, range_max]
  /odom             nav_msgs/Odometry   – robot pose in odom frame
  /goal_point       geometry_msgs/Point – target (x, y) in odom frame

Publication
-----------
  /vfh_command      Float64MultiArray   – [linear_vel, angular_vel] for act_node
  /goal_point_vis   PointStamped        – goal position in odom frame (for RViz)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, Float64MultiArray
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# VFH+ defaults  (all overridable as ROS parameters)
# ---------------------------------------------------------------------------
VFH_ALPHA           = 5      # degrees per histogram sector
VFH_THRESHOLD       = 0.20   # POD threshold: above → blocked
VFH_SMOOTH_WIN      = 3      # smoothing half-window (sectors each side)
VFH_D_MAX           = 3.5    # max sensor range (m)
VFH_ROBOT_RADIUS    = 0.18   # TurtleBot3 Burger radius (m)
VFH_SAFETY_DIST     = 0.75   # extra clearance (m)
VFH_A               = 1.0    # obstacle magnitude coefficient
VFH_B               = 1.0 / VFH_D_MAX

FIXED_LINEAR_VEL    = 0.15   # m/s  – constant forward speed (fixed-wing style)
MAX_ANGULAR_VEL     = 2.84   # rad/s
ARRIVAL_RADIUS      = 0.20   # m – goal considered reached inside this distance


# ===========================================================================
# Pure VFH+ implementation  (no ROS dependencies)
# ===========================================================================
class VFHPlanner:
    """
    Vector Field Histogram+ local planner.

    Parameters
    ----------
    alpha_deg      : sector width in degrees
    threshold      : POD value above which a sector is "blocked"
    smooth_window  : number of neighbouring sectors for smoothing (each side)
    d_max          : maximum sensor range (metres)
    robot_radius   : physical robot radius (metres)
    safety_dist    : extra clearance added to robot_radius (metres)
    a, b           : obstacle magnitude coefficients  h = (a - b·d)²
    fixed_speed    : constant forward velocity (m/s)
    max_angular    : angular velocity limit (rad/s)
    """

    def __init__(
        self,
        alpha_deg:     float = VFH_ALPHA,
        threshold:     float = VFH_THRESHOLD,
        smooth_window: int   = VFH_SMOOTH_WIN,
        d_max:         float = VFH_D_MAX,
        robot_radius:  float = VFH_ROBOT_RADIUS,
        safety_dist:   float = VFH_SAFETY_DIST,
        a:             float = VFH_A,
        b:             float = VFH_B,
        fixed_speed:   float = FIXED_LINEAR_VEL,
        max_angular:   float = MAX_ANGULAR_VEL,
    ) -> None:
        self.alpha        = math.radians(alpha_deg)
        self.threshold    = threshold
        self.smooth_w     = smooth_window
        self.d_max        = d_max
        self.a            = a
        self.b            = b
        self.fixed_speed  = fixed_speed
        self.max_angular  = max_angular
        self.r_enlarged   = robot_radius + safety_dist
        self.num_sectors  = max(1, round(2 * math.pi / self.alpha))

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
        goal_heading    : desired travel direction (radians, robot frame,
                          0 = straight ahead, positive = left)

        Returns
        -------
        (linear_vel, angular_vel)  in m/s and rad/s
        """
        hist        = self._build_histogram(ranges, angle_min, angle_increment)
        hist_smooth = self._smooth(hist)
        binary      = [1 if h > self.threshold else 0 for h in hist_smooth]
        valleys     = self._find_valleys(binary)
        best_dir    = self._select_direction(valleys, goal_heading)
        return self._to_velocities(best_dir)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------
    def _build_histogram(
        self,
        ranges:          list[float],
        angle_min:       float,
        angle_increment: float,
    ) -> list[float]:
        hist = [0.0] * self.num_sectors

        for i, d in enumerate(ranges):
            if d >= self.d_max:
                continue

            beam_angle = angle_min + i * angle_increment
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))
            sector     = int((beam_angle + math.pi) / self.alpha) % self.num_sectors
            magnitude  = (self.a - self.b * d) ** 2
            hist[sector] = max(hist[sector], magnitude)

        # Fixed cap instead of normalisation – avoids rescaling real obstacles
        # down below the threshold when a single far reading dominates.
        return [min(h, 1.0) for h in hist]

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------
    def _smooth(self, hist: list[float]) -> list[float]:
        n   = len(hist)
        out = [0.0] * n
        l   = self.smooth_w
        for k in range(n):
            total = sum(hist[(k + j) % n] for j in range(-l, l + 1))
            out[k] = total / (2 * l + 1)
        return out

    

    # ------------------------------------------------------------------
    # Valley detection
    # ------------------------------------------------------------------
    def _find_valleys(self, binary: list[int]) -> list[tuple[int, int]]:
        """Return list of (start_sector, end_sector) inclusive free valleys."""
        n        = len(binary)
        valleys  = []
        start    = None
        extended = binary + binary   # handle wrap-around

        for i in range(2 * n):
            if extended[i] == 0 and start is None:
                start = i
            elif extended[i] == 1 and start is not None:
                if start < n:
                    valleys.append((start % n, (i - 1) % n))
                start = None

        if start is not None and start < n:
            valleys.append((start % n, (2 * n - 1) % n))

        return valleys

    def _sector_to_angle(self, sector: int) -> float:
        angle = sector * self.alpha - math.pi + self.alpha / 2.0
        return math.atan2(math.sin(angle), math.cos(angle))

    # ------------------------------------------------------------------
    # Direction selection
    # ------------------------------------------------------------------
    def _select_direction(
        self,
        valleys:      list[tuple[int, int]],
        goal_heading: float,
    ) -> float | None:
        if not valleys:
            return None

        goal_sector = int((goal_heading + math.pi) / self.alpha) % self.num_sectors
        best_dir    = None
        best_cost   = float("inf")

        for v_start, v_end in valleys:
            if v_end >= v_start:
                sectors = range(v_start, v_end + 1)
            else:
                sectors = list(range(v_start, self.num_sectors)) + list(range(0, v_end + 1))

            for s in sectors:
                diff = abs(s - goal_sector)
                diff = min(diff, self.num_sectors - diff)
                if diff < best_cost:
                    best_cost = diff
                    best_dir  = self._sector_to_angle(s)

        return best_dir

    # ------------------------------------------------------------------
    # Velocity output  –  FIXED forward speed
    # ------------------------------------------------------------------
    def _to_velocities(self, direction: float | None) -> tuple[float, float]:
        """
        Fixed-wing style: linear velocity is always self.fixed_speed.
        Only angular velocity changes based on the chosen direction.
        If no valley is found, stop and spin to find an opening.
        """
        if direction is None:
            # Completely blocked – stop and turn hard to find a valley
            return 0.0, self.max_angular * 0.6

        # Proportional heading controller
        err         = math.atan2(math.sin(direction), math.cos(direction))
        angular_vel = float(np.clip(2.0 * err, -self.max_angular, self.max_angular))

        return self.fixed_speed, angular_vel


# ===========================================================================
# ROS 2 node
# ===========================================================================
class PlanNode(Node):
    """ROS 2 wrapper: goal-point tracking + VFH+ local avoidance."""

    def __init__(self) -> None:
        super().__init__("plan_node")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("alpha_deg",      VFH_ALPHA)
        self.declare_parameter("threshold",      VFH_THRESHOLD)
        self.declare_parameter("smooth_window",  VFH_SMOOTH_WIN)
        self.declare_parameter("d_max",          VFH_D_MAX)
        self.declare_parameter("robot_radius",   VFH_ROBOT_RADIUS)
        self.declare_parameter("safety_dist",    VFH_SAFETY_DIST)
        self.declare_parameter("vfh_a",          VFH_A)
        self.declare_parameter("vfh_b",          VFH_B)
        self.declare_parameter("fixed_speed",    FIXED_LINEAR_VEL)
        self.declare_parameter("max_angular",    MAX_ANGULAR_VEL)
        self.declare_parameter("arrival_radius", ARRIVAL_RADIUS)

        self._arrival_radius = self.get_parameter("arrival_radius").value

        self._planner = VFHPlanner(
            alpha_deg    = self.get_parameter("alpha_deg").value,
            threshold    = self.get_parameter("threshold").value,
            smooth_window= self.get_parameter("smooth_window").value,
            d_max        = self.get_parameter("d_max").value,
            robot_radius = self.get_parameter("robot_radius").value,
            safety_dist  = self.get_parameter("safety_dist").value,
            a            = self.get_parameter("vfh_a").value,
            b            = self.get_parameter("vfh_b").value,
            fixed_speed  = self.get_parameter("fixed_speed").value,
            max_angular  = self.get_parameter("max_angular").value,
        )

        # ---- robot state --------------------------------------------------
        self._robot_x:   float = 0.0
        self._robot_y:   float = 0.0
        self._robot_yaw: float = 0.0

        # ---- goal state ---------------------------------------------------
        self._goal_x:    float | None = None
        self._goal_y:    float | None = None
        self._goal_active: bool       = False

        # ---- scan state ---------------------------------------------------
        self._ranges:        list[float] | None = None
        self._angle_min:     float = 0.0               # LDS-01 default (not -pi)
        self._angle_inc:     float = math.radians(1.0)
        self._meta_received: bool  = False              # gate: wait for metadata

        # ---- QoS to match see_node BEST_EFFORT publisher ------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers --------------------------------------------------
        self._scan_sub = self.create_subscription(
            Float32MultiArray, "/processed_scan", self._scan_cb, sensor_qos,
        )
        self._meta_sub = self.create_subscription(
            Float64MultiArray, "/scan_metadata", self._meta_cb, 10,
        )
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10,
        )
        self._goal_sub = self.create_subscription(
            Point, "/goal_point", self._goal_cb, 10,
        )

        # ---- publishers ---------------------------------------------------
        self._cmd_pub      = self.create_publisher(Float64MultiArray, "/vfh_command",     10)
        self._goal_vis_pub = self.create_publisher(PointStamped,      "/goal_point_vis",  10)

        self.get_logger().info(
            "PlanNode ready  |  VFH+ (fixed-speed)  |  "
            "publish goal: ros2 topic pub /goal_point geometry_msgs/Point "
            "\"x: 2.0, y: 1.0, z: 0.0\""
        )

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------
    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._angle_min          = msg.data[0]
            self._angle_inc          = msg.data[2]
            self._planner.d_max      = msg.data[3]
            self._meta_received      = True

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

        q    = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)

        # Check arrival every odom update
        if self._goal_active:
            dx   = self._goal_x - self._robot_x
            dy   = self._goal_y - self._robot_y
            dist = math.hypot(dx, dy)
            if dist < self._arrival_radius:
                self.get_logger().info(
                    f"Goal reached  ({self._goal_x:.2f}, {self._goal_y:.2f})  "
                    f"dist={dist:.3f} m – publishing stop"
                )
                self._goal_active = False
                self._publish_command(0.0, 0.0)

    def _goal_cb(self, msg: Point) -> None:
        self._goal_x      = msg.x
        self._goal_y      = msg.y
        self._goal_active = True
        self.get_logger().info(
            f"New goal received: ({msg.x:.2f}, {msg.y:.2f})"
        )

        # Publish a PointStamped so RViz can display the goal in the odom frame
        vis = PointStamped()
        vis.header.stamp    = self.get_clock().now().to_msg()
        vis.header.frame_id = "odom"
        vis.point.x = msg.x
        vis.point.y = msg.y
        vis.point.z = 0.0
        self._goal_vis_pub.publish(vis)

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        """Main planning trigger – fires every time a new scan arrives."""
        self._ranges = list(msg.data)

        if not self._ranges:
            return

        # Gate: don't plan until beam geometry is known
        if not self._meta_received:
            self.get_logger().warn(
                "Waiting for /scan_metadata before planning...", throttle_duration_sec=2.0
            )
            return

        # If no goal is set or goal was reached, stop
        if not self._goal_active:
            self._publish_command(0.0, 0.0)
            return

        # Compute local goal heading from current pose + goal position
        dx           = self._goal_x - self._robot_x
        dy           = self._goal_y - self._robot_y
        world_bear   = math.atan2(dy, dx)
        goal_heading = math.atan2(
            math.sin(world_bear - self._robot_yaw),
            math.cos(world_bear - self._robot_yaw),
        )

        lin, ang = self._planner.compute(
            ranges          = self._ranges,
            angle_min       = self._angle_min,
            angle_increment = self._angle_inc,
            goal_heading    = goal_heading,
        )

        dist = math.hypot(dx, dy)
        self.get_logger().debug(
            f"goal_head={math.degrees(goal_heading):.1f}°  "
            f"dist={dist:.2f} m  v={lin:.3f}  ω={ang:.3f}"
        )

        self._publish_command(lin, ang)

    # -----------------------------------------------------------------------
    def _publish_command(self, linear: float, angular: float) -> None:
        msg      = Float64MultiArray()
        msg.data = [linear, angular]
        self._cmd_pub.publish(msg)


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