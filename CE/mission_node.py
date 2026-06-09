#!/usr/bin/env python3
"""
mission_node.py  –  Subsumption architecture mission controller.

Three behaviours, highest priority suppresses lower:

  Layer 3 (HIGH)   –  TRACK
      Active when /poacher_visible is True.
      Publishes the poacher's live position directly to /global_goal.
      Continuously updates LKP while visible.

  Layer 2 (MID)    –  GOTO_LKP
      Active immediately after the poacher disappears.
      Drives the drone to the last known position.
      Transitions to SEARCH once within arrival_radius of LKP.

  Layer 1 (LOW)    –  SEARCH
      Active once the drone has reached the LKP.
      Maintains a 2-D Gaussian diffusion PDF. Each tick the LiDAR
      scan is used to erase all PDF cells confirmed empty by line-of-
      sight (Bayesian update: searched and not found). The highest-
      probability remaining cell is sent as /global_goal (Stone's rule).

Publications
------------
  /global_goal        geometry_msgs/Point         → global_planner_node
  /pdf_map            nav_msgs/OccupancyGrid       → RViz (probability map)
  /mission_marker     visualization_msgs/Marker    → RViz (goal arrow)
  /mission_state      std_msgs/String              → TRACK/GOTO_LKP/SEARCH/INIT

Subscriptions
-------------
  /poacher_visible    std_msgs/Bool
  /poacher_odom       nav_msgs/Odometry
  /odom               nav_msgs/Odometry
  /processed_scan     std_msgs/Float32MultiArray   – for scan-erasure
  /scan_metadata      std_msgs/Float64MultiArray   – beam geometry

Parameters
----------
  diffusion_coeff     float   m²/s   spreading rate (default 1.0)
  search_radius       float   m      PDF grid half-size (default 8.0)
  grid_resolution     float   m/cell (default 0.25)
  arrival_radius      float   m      how close counts as arrived (default 0.5)
  pdf_publish_rate    float   Hz     (default 2.0)
  goal_publish_rate   float   Hz     (default 2.0)
  poacher_spawn_x/y   float   m      poacher spawn (default 2.0, 0.0)
  drone_spawn_x/y     float   m      drone spawn   (default -2.0, -0.5)
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from std_msgs.msg import Bool, Float64, Float32MultiArray, Float64MultiArray, String
from visualization_msgs.msg import Marker


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_TRACK  = 'TRACK'
STATE_SEARCH = 'SEARCH'
STATE_INIT   = 'INIT'        # no poacher data yet


# ─────────────────────────────────────────────────────────────────────────────
class MissionNode(Node):

    def __init__(self) -> None:
        super().__init__('mission_node')

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter('diffusion_coeff',  1.0)
        self.declare_parameter('search_radius',    8.0)
        self.declare_parameter('grid_resolution',  0.25)
        self.declare_parameter('arrival_radius',   0.5)
        self.declare_parameter('pdf_publish_rate', 2.0)
        self.declare_parameter('goal_publish_rate',2.0)
        self.declare_parameter('poacher_spawn_x',  2.0)
        self.declare_parameter('poacher_spawn_y',  0.0)
        self.declare_parameter('drone_spawn_x',   -2.0)
        self.declare_parameter('drone_spawn_y',   -0.5)

        self._D          = self.get_parameter('diffusion_coeff').value
        self._radius     = self.get_parameter('search_radius').value
        self._res        = self.get_parameter('grid_resolution').value
        self._arr_r      = self.get_parameter('arrival_radius').value
        pdf_hz           = self.get_parameter('pdf_publish_rate').value
        goal_hz          = self.get_parameter('goal_publish_rate').value

        # Offset to convert poacher local odom → drone world frame
        self._poacher_offset_x = (self.get_parameter('poacher_spawn_x').value
                                - self.get_parameter('drone_spawn_x').value)
        self._poacher_offset_y = (self.get_parameter('poacher_spawn_y').value
                                - self.get_parameter('drone_spawn_y').value)

        # ── state ─────────────────────────────────────────────────────────────
        self._visible:          bool  = False
        self._drone_x:          float = 0.0
        self._drone_y:          float = 0.0
        self._poacher_x:        float = 0.0
        self._poacher_y:        float = 0.0
        self._poacher_received: bool  = False

        # Last known position and the time it was observed
        self._lkp_x:    float = 0.0
        self._lkp_y:    float = 0.0
        self._lkp_time: float = 0.0   # wall-clock seconds

        # PDF grid (lazily initialised when first LKP is set)
        self._pdf:         np.ndarray | None = None
        self._pdf_origin_x: float = 0.0
        self._pdf_origin_y: float = 0.0
        self._pdf_rows:     int   = 0
        self._pdf_cols:     int   = 0

        # Current goal being sent to the global planner
        self._current_goal: tuple | None = None   # (x, y)
        self._mission_state: str = STATE_INIT

        # Scan state (for PDF erasure)
        self._ranges:        list[float] = []
        self._angle_min:     float = 0.0
        self._angle_inc:     float = math.radians(1.0)
        self._drone_yaw:     float = 0.0
        self._scan_ready:    bool  = False

        # ── QoS ──────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Bool,              '/poacher_visible', self._visible_cb,  10)
        self.create_subscription(Odometry,          '/odom',            self._drone_cb,    sensor_qos)
        self.create_subscription(Odometry,          '/poacher_odom',    self._poacher_cb,  sensor_qos)
        self.create_subscription(Float32MultiArray, '/processed_scan',  self._scan_cb,     sensor_qos)
        self.create_subscription(Float64MultiArray, '/scan_metadata',   self._meta_cb,     10)

        # ── publishers ───────────────────────────────────────────────────────
        self._goal_pub    = self.create_publisher(Point,        '/global_goal',     reliable_qos)
        self._pdf_pub     = self.create_publisher(OccupancyGrid,'/pdf_map',         reliable_qos)
        self._marker_pub  = self.create_publisher(Marker,       '/mission_marker',  10)
        self._state_pub   = self.create_publisher(String,       '/mission_state',   10)

        # ── timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / pdf_hz,  self._pdf_timer_cb)
        self.create_timer(1.0 / goal_hz, self._goal_timer_cb)

        self.get_logger().info('MissionNode ready')

    # ─────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ─────────────────────────────────────────────────────────────────────────

    def _visible_cb(self, msg: Bool) -> None:
        was_visible = self._visible
        self._visible = msg.data

        if self._visible:
            # Poacher just became visible (or stays visible): update LKP
            if self._poacher_received:
                self._lkp_x    = self._poacher_x
                self._lkp_y    = self._poacher_y
                self._lkp_time = time.monotonic()

        elif was_visible and not self._visible:
            # Poacher just disappeared: freeze LKP and initialise PDF
            self.get_logger().info(
                f'Poacher lost at ({self._lkp_x:.2f}, {self._lkp_y:.2f})'
                f' – switching to SEARCH'
            )
            self._init_pdf(self._lkp_x, self._lkp_y)

    def _drone_cb(self, msg: Odometry) -> None:
        self._drone_x = msg.pose.pose.position.x
        self._drone_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._drone_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        self._ranges = list(msg.data)
        self._scan_ready = bool(self._ranges)

    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 3:
            self._angle_min = msg.data[0]
            self._angle_inc = msg.data[2]

    def _poacher_cb(self, msg: Odometry) -> None:
        self._poacher_x = msg.pose.pose.position.x + self._poacher_offset_x
        self._poacher_y = msg.pose.pose.position.y + self._poacher_offset_y

        if not self._poacher_received:
            # First message ever — seed LKP so SEARCH has a valid starting point
            # even if the poacher was never explicitly visible first.
            self._lkp_x    = self._poacher_x
            self._lkp_y    = self._poacher_y
            self._lkp_time = time.monotonic()
            self._init_pdf(self._lkp_x, self._lkp_y)
            self.get_logger().info(
                f'LKP seeded from first odom: '
                f'({self._lkp_x:.2f}, {self._lkp_y:.2f})'
            )

        self._poacher_received = True

    # ─────────────────────────────────────────────────────────────────────────
    # PDF management
    # ─────────────────────────────────────────────────────────────────────────

    def _init_pdf(self, cx: float, cy: float) -> None:
        """Create a fresh Gaussian PDF grid centred on (cx, cy)."""
        half = self._radius
        self._pdf_origin_x = cx - half
        self._pdf_origin_y = cy - half
        size = 2.0 * half
        self._pdf_cols = max(4, int(size / self._res))
        self._pdf_rows = max(4, int(size / self._res))

        # Coordinates of each cell centre in world frame
        xs = self._pdf_origin_x + (np.arange(self._pdf_cols) + 0.5) * self._res
        ys = self._pdf_origin_y + (np.arange(self._pdf_rows) + 0.5) * self._res
        XX, YY = np.meshgrid(xs, ys)   # shape (rows, cols)

        # Initial Gaussian: σ = 0.5 m (tight – just lost him)
        sigma_init = 0.5
        dist2 = (XX - cx) ** 2 + (YY - cy) ** 2
        pdf = np.exp(-dist2 / (2.0 * sigma_init ** 2))
        self._pdf = pdf / pdf.sum()   # normalise to probability mass

    def _diffuse_pdf(self) -> None:
        """
        Spread the PDF using discrete Gaussian diffusion.
        σ grows as sqrt(2 * D * Δt) since the LKP was set.
        We approximate this by convolving with a Gaussian kernel each tick.
        """
        if self._pdf is None:
            return
        dt   = time.monotonic() - self._lkp_time
        sigma_cells = math.sqrt(2.0 * self._D * dt) / self._res
        sigma_cells = max(0.5, sigma_cells)

        # Build a small Gaussian kernel
        ksize = max(3, int(6 * sigma_cells) | 1)   # odd, ≥3
        half  = ksize // 2
        ax    = np.arange(-half, half + 1)
        kernel_1d = np.exp(-ax ** 2 / (2.0 * sigma_cells ** 2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Convolve (separable would be faster, but grid is small)
        from scipy.ndimage import convolve
        diffused = convolve(self._pdf, kernel_2d, mode='reflect')
        self._pdf = diffused / diffused.sum()   # keep normalised

    def _mark_visited(self, drone_x: float, drone_y: float) -> None:
        """Zero out PDF cells within arrival_radius of the drone."""
        if self._pdf is None:
            return
        col0 = int((drone_x - self._pdf_origin_x) / self._res)
        row0 = int((drone_y - self._pdf_origin_y) / self._res)
        r_cells = max(1, int(self._arr_r / self._res))

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                r = row0 + dr
                c = col0 + dc
                if 0 <= r < self._pdf_rows and 0 <= c < self._pdf_cols:
                    dist = math.hypot(dr, dc) * self._res
                    if dist <= self._arr_r:
                        self._pdf[r, c] = 0.0

        total = self._pdf.sum()
        if total > 1e-9:
            self._pdf /= total   # re-normalise after zeroing

    def _erase_visible_sector(self) -> None:
        """
        Bayesian update: for every LiDAR beam, walk along it and zero any
        PDF cell the beam confirms is empty (beam reached that far without
        hitting anything). Cells beyond the beam's measured range are left
        alone — something may be hiding behind the obstacle.

        This collapses the PDF quickly when the drone has clear sightlines
        across the search area, rather than only erasing cells it physically
        visits.
        """
        if self._pdf is None or not self._scan_ready:
            return

        changed = False
        n_beams = len(self._ranges)

        for i, r in enumerate(self._ranges):
            if r <= 0.0:
                continue   # invalid beam

            # World-frame direction of this beam
            beam_angle = self._drone_yaw + self._angle_min + i * self._angle_inc

            # Step along the beam in cell-sized increments
            steps = int(r / self._res)
            for s in range(1, steps + 1):
                wx = self._drone_x + s * self._res * math.cos(beam_angle)
                wy = self._drone_y + s * self._res * math.sin(beam_angle)

                col = int((wx - self._pdf_origin_x) / self._res)
                row = int((wy - self._pdf_origin_y) / self._res)

                if 0 <= row < self._pdf_rows and 0 <= col < self._pdf_cols:
                    if self._pdf[row, col] > 0.0:
                        self._pdf[row, col] = 0.0
                        changed = True

        if changed:
            total = self._pdf.sum()
            if total > 1e-9:
                self._pdf /= total
        """Return world (x, y) of the highest-probability cell."""
        if self._pdf is None:
            return None
        idx = np.argmax(self._pdf)
        row, col = divmod(int(idx), self._pdf_cols)
        wx = self._pdf_origin_x + (col + 0.5) * self._res
        wy = self._pdf_origin_y + (row + 0.5) * self._res
        return (wx, wy)

    # ─────────────────────────────────────────────────────────────────────────
    # Timers
    # ─────────────────────────────────────────────────────────────────────────

    def _goal_timer_cb(self) -> None:
        """
        Subsumption arbiter — three layers, highest wins:
          Layer 3  TRACK     – poacher visible
          Layer 2  GOTO_LKP  – heading to last known position
          Layer 1  SEARCH    – PDF-guided search
        """
        if not self._poacher_received:
            self._mission_state = STATE_INIT
            self._publish_state()
            return

        if self._visible:
            # ── Layer 3: TRACK ───────────────────────────────────────────────
            self._mission_state = STATE_TRACK
            goal = (self._poacher_x, self._poacher_y)

        else:
            # ── Layer 1: SEARCH ──────────────────────────────────────────────
            self._mission_state = STATE_SEARCH

            # Diffuse PDF over elapsed time
            self._diffuse_pdf()

            # Erase cells confirmed empty by current scan (fast collapse)
            self._erase_visible_sector()

            # Also erase cells immediately around the drone
            self._mark_visited(self._drone_x, self._drone_y)

            # Hold LKP goal until drone arrives — then let PDF pick next target
            dist_to_lkp = math.hypot(
                self._drone_x - self._lkp_x,
                self._drone_y - self._lkp_y,
            )
            if dist_to_lkp > self._arr_r:
                goal = (self._lkp_x, self._lkp_y)
            else:
                goal = self._best_search_goal()
                if goal is None:
                    goal = (self._lkp_x, self._lkp_y)

        self._current_goal = goal
        self._publish_goal(goal)
        self._publish_goal_marker(goal)
        self._publish_state()

    def _pdf_timer_cb(self) -> None:
        """Publish the PDF as an OccupancyGrid for RViz."""
        if self._pdf is None or self._mission_state == STATE_TRACK:
            return
        self._publish_pdf()

    # ─────────────────────────────────────────────────────────────────────────
    # Publishers
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_goal(self, goal: tuple) -> None:
        pt = Point()
        pt.x, pt.y, pt.z = goal[0], goal[1], 0.0
        self._goal_pub.publish(pt)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self._mission_state
        self._state_pub.publish(msg)

    def _publish_goal_marker(self, goal: tuple) -> None:
        """Arrow pointing from drone to current goal, colour-coded by state."""
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.ns, m.id        = 'mission_goal', 0
        m.type            = Marker.ARROW
        m.action          = Marker.ADD
        m.scale.x = 0.08   # shaft diameter
        m.scale.y = 0.16   # head diameter
        m.scale.z = 0.0

        if self._mission_state == STATE_TRACK:
            # Green – tracking live poacher
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.2, 0.9
        else:
            # Yellow – searching (includes heading to LKP)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 0.9

        def pt(x, y):
            p = Point()
            p.x, p.y, p.z = x, y, 0.2
            return p

        m.points.append(pt(self._drone_x, self._drone_y))
        m.points.append(pt(goal[0],       goal[1]))
        self._marker_pub.publish(m)

        # Also a sphere at the goal itself
        s = Marker()
        s.header          = m.header
        s.ns, s.id        = 'mission_goal', 1
        s.type            = Marker.SPHERE
        s.action          = Marker.ADD
        s.pose.position.x = goal[0]
        s.pose.position.y = goal[1]
        s.pose.position.z = 0.2
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.25
        s.color           = m.color
        self._marker_pub.publish(s)

    def _publish_pdf(self) -> None:
        """Publish the probability density as an OccupancyGrid (0–100)."""
        grid = OccupancyGrid()
        grid.header.stamp    = self.get_clock().now().to_msg()
        grid.header.frame_id = 'odom'

        info = MapMetaData()
        info.resolution      = self._res
        info.width           = self._pdf_cols
        info.height          = self._pdf_rows
        info.origin.position.x = self._pdf_origin_x
        info.origin.position.y = self._pdf_origin_y
        info.origin.orientation.w = 1.0
        grid.info = info

        # Normalise to 0–100 for OccupancyGrid
        flat  = self._pdf.flatten()
        mx    = flat.max()
        if mx > 0:
            scaled = (flat / mx * 99.0).astype(np.int8)
        else:
            scaled = np.zeros_like(flat, dtype=np.int8)

        grid.data = scaled.tolist()
        self._pdf_pub.publish(grid)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()