#!/usr/bin/env python3
"""
poacher_node.py  –  Simulated poacher for TurtleBot3 / Gazebo.

Motion
------
  The poacher is a kinematic point that follows a waypoint queue using a
  Tangent Bug algorithm for obstacle avoidance:

    1. DIRECT mode  – move straight toward the current waypoint.
    2. BOUNDARY mode – an obstacle blocks the straight line; hug its boundary
       (always choosing the side that reduces distance-to-goal) until the
       direct path is clear again and the poacher is closer to the goal than
       when it started hugging.

  This avoids the "stuck in a corner" failure of pure repulsion while staying
  simple and predictable.

RViz2 visualisation
-------------------
  /poacher_marker       Marker  SPHERE  – current position (solid red)
  /poacher_path_marker  Marker  LINE_STRIP – planned path to next waypoint,
                                updated every tick to show the live bug route

Topics
------
  Subscriptions
    /gazebo/model_states   gazebo_msgs/ModelStates   obstacle positions
    /poacher_waypoints     geometry_msgs/PoseArray   runtime waypoint override

  Publications
    /poacher_odom          nav_msgs/Odometry
    /poacher_marker        visualization_msgs/Marker  (sphere)
    /poacher_path_marker   visualization_msgs/Marker  (line strip)

Parameters
----------
  waypoints        str    semicolon-separated "x,y" pairs for the default path
                          e.g. "0.0,0.0; 2.0,1.0; 4.0,-1.0"
                          The poacher spawns at the FIRST point.
  speed            float  m/s  travel speed              (default 0.15)
  max_speed        float  m/s  hard upper clamp          (default 0.15)
  update_rate      float  Hz                             (default 20.0)
  waypoint_radius  float  m    arrival threshold         (default 0.25)
  obstacle_radius  float  m    bounding circle / model   (default 0.35)
  avoid_dist       float  m    detection horizon         (default 0.8)
  obstacle_models  str    comma-separated name substrings
                          (default "wall,box,cylinder,pillar,obstacle")
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseArray, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

try:
    from gazebo_msgs.msg import ModelStates
    _GAZEBO_AVAILABLE = True
except ImportError:
    _GAZEBO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SPEED           = 0.15
DEFAULT_MAX_SPEED       = 0.15
DEFAULT_RATE            = 20.0
DEFAULT_WP_RADIUS       = 0.25
DEFAULT_OBSTACLE_RADIUS = 0.35
DEFAULT_AVOID_DIST      = 0.8
DEFAULT_OBSTACLE_NAMES  = "wall,box,cylinder,pillar,obstacle"
DEFAULT_WAYPOINTS       = ""        # empty → stay at origin

COLOUR_SPHERE = (1.0, 0.0, 0.0, 1.0)
COLOUR_PATH   = (1.0, 0.35, 0.35, 0.85)


# ---------------------------------------------------------------------------
def _parse_waypoints(param: str) -> list[tuple[float, float]]:
    """Parse "x,y; x,y; ..." into [(x,y), ...].  Returns [] on empty input."""
    result = []
    for token in param.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            continue
        try:
            result.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return result


# ---------------------------------------------------------------------------
class PoacherNode(Node):

    # Bug states
    _DIRECT   = "DIRECT"
    _BOUNDARY = "BOUNDARY"

    def __init__(self) -> None:
        super().__init__("poacher_node")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("waypoints",       DEFAULT_WAYPOINTS)
        self.declare_parameter("speed",           DEFAULT_SPEED)
        self.declare_parameter("max_speed",       DEFAULT_MAX_SPEED)
        self.declare_parameter("update_rate",     DEFAULT_RATE)
        self.declare_parameter("waypoint_radius", DEFAULT_WP_RADIUS)
        self.declare_parameter("obstacle_radius", DEFAULT_OBSTACLE_RADIUS)
        self.declare_parameter("avoid_dist",      DEFAULT_AVOID_DIST)
        self.declare_parameter("obstacle_models", DEFAULT_OBSTACLE_NAMES)

        self._speed      = min(
            self.get_parameter("speed").value,
            self.get_parameter("max_speed").value,
        )
        self._max_speed  = self.get_parameter("max_speed").value
        self._rate       = self.get_parameter("update_rate").value
        self._wp_radius  = self.get_parameter("waypoint_radius").value
        self._obs_radius = self.get_parameter("obstacle_radius").value
        self._avoid_dist = self.get_parameter("avoid_dist").value
        self._obs_tags   = [
            s.strip()
            for s in self.get_parameter("obstacle_models").value.split(",")
            if s.strip()
        ]

        # ---- waypoints ----------------------------------------------------
        raw = self.get_parameter("waypoints").value
        self._waypoints: list[tuple[float, float]] = _parse_waypoints(raw)
        self._wp_index:  int = 0

        # Spawn at first waypoint if provided, else origin
        if self._waypoints:
            self._x, self._y = self._waypoints[0]
            self._wp_index   = 1          # first WP is spawn; head for second
        else:
            self._x, self._y = 0.0, 0.0

        self._yaw: float = 0.0

        # ---- bug algorithm state ------------------------------------------
        self._bug_state: str   = self._DIRECT
        # Distance to goal when boundary-following started
        self._bug_start_dist:  float = float("inf")
        # Which side to hug: +1 = left (CCW), -1 = right (CW)
        self._bug_side:        int   = 1
        # Breadcrumb trail of boundary positions for the path marker
        self._bug_trail:       list[tuple[float, float]] = []

        # ---- obstacle centres from Gazebo ---------------------------------
        self._obstacle_centres: list[tuple[float, float]] = []

        # ---- QoS ----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers --------------------------------------------------
        if _GAZEBO_AVAILABLE:
            self.create_subscription(
                ModelStates, "/gazebo/model_states",
                self._model_states_cb, sensor_qos,
            )
        else:
            self.get_logger().warn(
                "gazebo_msgs not available – obstacle avoidance disabled."
            )

        self.create_subscription(
            PoseArray, "/poacher_waypoints", self._waypoints_cb, 10,
        )

        # ---- publishers ---------------------------------------------------
        self._odom_pub   = self.create_publisher(Odometry, "/poacher_odom",        10)
        self._sphere_pub = self.create_publisher(Marker,   "/poacher_marker",      10)
        self._path_pub   = self.create_publisher(Marker,   "/poacher_path_marker", 10)

        # ---- timer --------------------------------------------------------
        self._dt = 1.0 / self._rate
        self.create_timer(self._dt, self._update)

        self.get_logger().info(
            f"PoacherNode ready  |  speed={self._speed} m/s  "
            f"max={self._max_speed} m/s  "
            f"start=({self._x:.2f}, {self._y:.2f})  "
            f"{len(self._waypoints)} waypoint(s)\n"
            "  Runtime override:\n"
            "  ros2 topic pub --times 1 /poacher_waypoints "
            "geometry_msgs/msg/PoseArray "
            '"{poses: [{position: {x: 1.0, y: 0.0}}, {position: {x: 3.0, y: 2.0}}]}"'
        )

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------
    def _model_states_cb(self, msg: "ModelStates") -> None:
        centres = []
        for name, pose in zip(msg.name, msg.pose):
            nl = name.lower()
            if "ground" in nl or "turtlebot" in nl or "burger" in nl:
                continue
            if not any(tag in nl for tag in self._obs_tags):
                continue
            centres.append((pose.position.x, pose.position.y))
        self._obstacle_centres = centres

    def _waypoints_cb(self, msg: PoseArray) -> None:
        """Runtime override: spawn at first pose, then follow the rest."""
        new_wps = [(p.position.x, p.position.y) for p in msg.poses]
        if not new_wps:
            self.get_logger().warn("Empty waypoint list – ignored.")
            return
        # Teleport to first point and start moving toward second
        self._x, self._y = new_wps[0]
        self._waypoints  = new_wps
        self._wp_index   = 1
        self._reset_bug()
        self.get_logger().info(
            f"Waypoints updated: {len(new_wps)} points  "
            f"spawn=({self._x:.2f}, {self._y:.2f})"
        )

    # -----------------------------------------------------------------------
    # Bug helpers
    # -----------------------------------------------------------------------
    def _reset_bug(self) -> None:
        self._bug_state      = self._DIRECT
        self._bug_start_dist = float("inf")
        self._bug_side       = 1
        self._bug_trail      = []

    def _nearest_obstacle(self) -> tuple[float, float, float] | None:
        """Return (ox, oy, dist_to_surface) of the closest obstacle, or None."""
        best = None
        best_d = float("inf")
        for (ox, oy) in self._obstacle_centres:
            d = math.hypot(self._x - ox, self._y - oy) - self._obs_radius
            if d < best_d:
                best_d = d
                best   = (ox, oy, d)
        return best

    def _path_blocked(self, tx: float, ty: float) -> tuple[bool, tuple | None]:
        """
        Check whether any obstacle centre sits closer than avoid_dist to the
        line segment from current pos to (tx, ty).
        Returns (blocked, nearest_obstacle_or_None).
        """
        dx = tx - self._x
        dy = ty - self._y
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return False, None

        ux, uy = dx / seg_len, dy / seg_len   # unit vec along segment

        nearest = None
        nearest_d = float("inf")

        for (ox, oy) in self._obstacle_centres:
            # Vector from current pos to obstacle centre
            ex = ox - self._x
            ey = oy - self._y
            # Project onto segment
            t = max(0.0, min(seg_len, ex * ux + ey * uy))
            # Closest point on segment
            cx = self._x + t * ux
            cy = self._y + t * uy
            # Distance from that point to obstacle surface
            surf_d = math.hypot(ox - cx, oy - cy) - self._obs_radius
            if surf_d < self._avoid_dist:
                if surf_d < nearest_d:
                    nearest_d = surf_d
                    nearest = (ox, oy, surf_d)

        return (nearest is not None), nearest

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    def _update(self) -> None:
        vx, vy = self._compute_velocity()
        self._x   += vx * self._dt
        self._y   += vy * self._dt
        if vx != 0.0 or vy != 0.0:
            self._yaw = math.atan2(vy, vx)

        self._publish_odom(vx, vy)
        self._publish_sphere_marker()
        self._publish_path_marker()

    def _compute_velocity(self) -> tuple[float, float]:
        # No goal → stand still
        if not self._waypoints or self._wp_index >= len(self._waypoints):
            return 0.0, 0.0

        wx, wy = self._waypoints[self._wp_index]

        # ---- arrival check ------------------------------------------------
        dist_to_wp = math.hypot(wx - self._x, wy - self._y)
        if dist_to_wp < self._wp_radius:
            self._wp_index += 1
            self._reset_bug()
            if self._wp_index >= len(self._waypoints):
                self.get_logger().info("All waypoints reached – poacher stopped.")
                return 0.0, 0.0
            wx, wy = self._waypoints[self._wp_index]
            dist_to_wp = math.hypot(wx - self._x, wy - self._y)
            self.get_logger().info(
                f"Waypoint {self._wp_index}/{len(self._waypoints)} "
                f"-> ({wx:.2f}, {wy:.2f})"
            )

        # ---- Tangent Bug state machine ------------------------------------
        if self._bug_state == self._DIRECT:
            blocked, blocker = self._path_blocked(wx, wy)
            if not blocked:
                # Straight shot – head directly for the waypoint
                return self._vel_toward(wx, wy)
            else:
                # Enter boundary-following mode
                self._bug_state      = self._BOUNDARY
                self._bug_start_dist = dist_to_wp
                self._bug_trail      = [(self._x, self._y)]
                # Decide which side to hug: pick the side that most quickly
                # opens the path (left vs right tangent point)
                ox, oy, _ = blocker
                self._bug_side = self._choose_side(wx, wy, ox, oy)
                self.get_logger().info(
                    f"BUG: entering BOUNDARY  "
                    f"side={'LEFT' if self._bug_side > 0 else 'RIGHT'}  "
                    f"dist={dist_to_wp:.2f} m"
                )
                return self._boundary_vel(ox, oy)

        else:  # BOUNDARY
            # Check if we can leave boundary mode:
            #   1. Direct path to waypoint is clear, AND
            #   2. We are strictly closer than when we started hugging
            blocked, blocker = self._path_blocked(wx, wy)
            if not blocked and dist_to_wp < self._bug_start_dist - self._wp_radius:
                self.get_logger().info(
                    f"BUG: leaving BOUNDARY  dist={dist_to_wp:.2f} m"
                )
                self._reset_bug()
                return self._vel_toward(wx, wy)

            # Still hugging – stay tangent to nearest obstacle
            nb = self._nearest_obstacle()
            if nb is None:
                # Lost the obstacle (shouldn't happen often) – try direct
                self._reset_bug()
                return self._vel_toward(wx, wy)

            ox, oy, surf_d = nb
            self._bug_trail.append((self._x, self._y))
            return self._boundary_vel(ox, oy)

    def _vel_toward(self, tx: float, ty: float) -> tuple[float, float]:
        """Unit vector toward (tx, ty) scaled to speed."""
        dx, dy = tx - self._x, ty - self._y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0, 0.0
        s = min(self._speed, self._max_speed)
        return dx / d * s, dy / d * s

    def _choose_side(self, wx: float, wy: float,
                     ox: float, oy: float) -> int:
        """
        Returns +1 (left/CCW) or -1 (right/CW).
        Picks the side whose tangent point is angularly closer to the goal.
        """
        # Vector from poacher to obstacle centre
        dx_o = ox - self._x
        dy_o = oy - self._y
        d_o  = math.hypot(dx_o, dy_o)
        if d_o < 1e-6:
            return 1

        # Tangent half-angle
        sin_t = min(self._obs_radius / d_o, 1.0)
        theta = math.asin(sin_t)

        # Obstacle bearing
        bear_o = math.atan2(dy_o, dx_o)
        # Goal bearing
        bear_g = math.atan2(wy - self._y, wx - self._x)

        # Left tangent bearing
        left_bear  = bear_o + (math.pi / 2 - theta)
        # Right tangent bearing
        right_bear = bear_o - (math.pi / 2 - theta)

        def ang_diff(a, b):
            d = (a - b + math.pi) % (2 * math.pi) - math.pi
            return abs(d)

        return 1 if ang_diff(left_bear, bear_g) < ang_diff(right_bear, bear_g) else -1

    def _boundary_vel(self, ox: float, oy: float) -> tuple[float, float]:
        """
        Move tangentially around the obstacle at (ox, oy).
        _bug_side = +1 → CCW (left),  -1 → CW (right).
        """
        # Vector from obstacle to poacher (outward radial)
        rx = self._x - ox
        ry = self._y - oy
        r  = math.hypot(rx, ry)
        if r < 1e-6:
            # Exactly on centre – nudge outward
            rx, ry, r = 1.0, 0.0, 1.0

        # Normalise
        rx /= r
        ry /= r

        # Desired orbit radius = obstacle radius + small clearance
        orbit_r = self._obs_radius + 0.15

        # Radial correction: push poacher toward the orbit radius
        radial_err  = r - orbit_r
        radial_gain = 1.5
        rad_vx = -rx * radial_err * radial_gain
        rad_vy = -ry * radial_err * radial_gain

        # Tangential velocity (perpendicular to radial, in chosen direction)
        # CCW (+1): tangent = (-ry,  rx)
        # CW  (-1): tangent = ( ry, -rx)
        tx = -ry * self._bug_side
        ty =  rx * self._bug_side

        s = min(self._speed, self._max_speed)
        cvx = tx * s + rad_vx
        cvy = ty * s + rad_vy

        # Clamp to max_speed
        cn = math.hypot(cvx, cvy)
        if cn > self._max_speed:
            cvx = cvx / cn * self._max_speed
            cvy = cvy / cn * self._max_speed

        return cvx, cvy

    # -----------------------------------------------------------------------
    # Publishers
    # -----------------------------------------------------------------------
    def _publish_odom(self, vx: float, vy: float) -> None:
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id  = "poacher"

        msg.pose.pose.position.x  = self._x
        msg.pose.pose.position.y  = self._y
        msg.pose.pose.position.z  = 0.0
        half = self._yaw / 2.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)

        msg.twist.twist.linear.x  = math.hypot(vx, vy)
        msg.twist.twist.linear.y  = 0.0
        msg.twist.twist.angular.z = 0.0
        self._odom_pub.publish(msg)

    def _publish_sphere_marker(self) -> None:
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "odom"
        m.ns, m.id        = "poacher", 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = self._x
        m.pose.position.y = self._y
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        r, g, b, a = COLOUR_SPHERE
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        self._sphere_pub.publish(m)

    def _publish_path_marker(self) -> None:
        """
        Line strip showing:
          current pos → (bug trail if in BOUNDARY mode) → remaining waypoints
        """
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "odom"
        m.ns, m.id        = "poacher_path", 1
        m.type            = Marker.LINE_STRIP
        m.action          = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        r, g, b, a = COLOUR_PATH
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a

        def pt(x, y):
            p = Point()
            p.x, p.y, p.z = x, y, 0.05
            return p

        # Current position
        m.points.append(pt(self._x, self._y))

        # If hugging an obstacle, show the live boundary trail
        if self._bug_state == self._BOUNDARY:
            for (bx, by) in self._bug_trail[-30:]:   # cap trail length
                m.points.append(pt(bx, by))

        # Remaining waypoints
        for wx, wy in self._waypoints[self._wp_index:]:
            m.points.append(pt(wx, wy))

        if len(m.points) <= 1:
            m.action = Marker.DELETE

        self._path_pub.publish(m)


# ---------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoacherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()