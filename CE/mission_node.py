#!/usr/bin/env python3
"""
mission_node.py  –  Subsumption architecture mission controller.

Layer 2 (HIGH)  TRACK    – poacher visible  → follow live position
Layer 1 (LOW)   SEARCH   – poacher hidden:
    Phase A  PDF-guided: Gaussian diffusion from LKP, erased by current
             scan within max_range. Goal = highest remaining probability
             cell. Holds each goal until arrival before picking next.
    Phase B  Waypoint:   PDF max < threshold → build obstacle-fringe
             waypoint list from costmap, navigate each in order.

Publications
------------
  /global_goal        geometry_msgs/Point
  /pdf_map            nav_msgs/OccupancyGrid
  /mission_marker     visualization_msgs/Marker
  /mission_state      std_msgs/String

Subscriptions
-------------
  /poacher_visible    std_msgs/Bool
  /poacher_odom       nav_msgs/Odometry
  /odom               nav_msgs/Odometry
  /costmap            nav_msgs/OccupancyGrid
  /processed_scan     std_msgs/Float32MultiArray
  /scan_metadata      std_msgs/Float64MultiArray

Parameters
----------
  arrival_radius          float  m      (default 0.5)
  goal_publish_rate       float  Hz     (default 2.0)
  diffusion_coeff         float  m²/s   (default 0.01)
  search_radius           float  m      PDF half-size (default 4.0)
  grid_resolution         float  m/cell (default 0.1)
  max_range               float  m      scan mask range (default 3.5)
  pdf_exhausted_threshold float         (default 0.005)
  poacher_spawn_x/y       float  m      (default 2.0 / 0.0)
  drone_spawn_x/y         float  m      (default -2.0 / -0.5)
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.ndimage import convolve, binary_dilation

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool, String, Float32MultiArray, Float64MultiArray
from visualization_msgs.msg import Marker

# ── Arena hard bounds ─────────────────────────────────────────────────────────
ARENA_X_MIN, ARENA_X_MAX =  0.0,  3.5
ARENA_Y_MIN, ARENA_Y_MAX = -0.5,  2.0

# ── Costmap thresholds ────────────────────────────────────────────────────────
COST_OBS_MIN = 50.0

# ── States ────────────────────────────────────────────────────────────────────
STATE_INIT     = 'INIT'
STATE_TRACK    = 'TRACK'
STATE_GOTO     = 'GOTO_LKP'
STATE_PDF      = 'PDF_SEARCH'
STATE_WAYPOINT = 'WAYPOINT_SEARCH'


class MissionNode(Node):

    def __init__(self) -> None:
        super().__init__('mission_node')

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter('arrival_radius',          0.5)
        self.declare_parameter('goal_publish_rate',       2.0)
        self.declare_parameter('diffusion_coeff',         0.004)
        self.declare_parameter('search_radius',           2.5)
        self.declare_parameter('grid_resolution',         0.1)
        self.declare_parameter('max_range',               3.5)
        self.declare_parameter('scan_clear_range',        1.0)
        self.declare_parameter('poacher_spawn_x',         2.0)
        self.declare_parameter('poacher_spawn_y',         0.0)
        self.declare_parameter('drone_spawn_x',          -2.0)
        self.declare_parameter('drone_spawn_y',          -0.5)

        self._arr_r     = self.get_parameter('arrival_radius').value
        goal_hz         = self.get_parameter('goal_publish_rate').value
        self._D         = self.get_parameter('diffusion_coeff').value
        self._radius    = self.get_parameter('search_radius').value
        self._res       = self.get_parameter('grid_resolution').value
        self._max_range        = self.get_parameter('max_range').value
        self._scan_clear_range = self.get_parameter('scan_clear_range').value

        self._poacher_offset_x = (self.get_parameter('poacher_spawn_x').value
                                 - self.get_parameter('drone_spawn_x').value)
        self._poacher_offset_y = (self.get_parameter('poacher_spawn_y').value
                                 - self.get_parameter('drone_spawn_y').value)

        # ── robot state ───────────────────────────────────────────────────────
        self._visible          = False
        self._drone_x          = 0.0
        self._drone_y          = 0.0
        self._drone_yaw        = 0.0
        self._poacher_x        = 0.0
        self._poacher_y        = 0.0
        self._poacher_received = False

        self._lkp_x    = 0.0
        self._lkp_y    = 0.0
        self._lkp_time = time.monotonic()

        # ── visibility debounce ───────────────────────────────────────────────
        self._vis_buffer:     list[bool] = []
        self._VIS_DEBOUNCE_N: int        = 5

        # ── mission state ─────────────────────────────────────────────────────
        self._mission_state = STATE_INIT
        self._in_search     = False   # True once arrived at LKP

        # ── PDF grid ──────────────────────────────────────────────────────────
        self._pdf:           np.ndarray | None = None
        self._pdf_cleared:   np.ndarray | None = None  # permanent cleared mask
        self._pdf_origin_x:  float = 0.0
        self._pdf_origin_y:  float = 0.0
        self._pdf_rows:      int   = 0
        self._pdf_cols:      int   = 0
        self._pdf_goal:      tuple | None = None
        self._last_diffuse_t: float = 0.0   # current committed PDF goal

        # ── scan state ───────────────────────────────────────────────────────
        self._ranges:     list[float] = []
        self._angle_min:  float = 0.0
        self._angle_inc:  float = math.radians(1.0)
        self._scan_ready: bool  = False

        # ── costmap ───────────────────────────────────────────────────────────
        self._costmap_data:     np.ndarray | None = None
        self._costmap_origin_x: float = -5.0
        self._costmap_origin_y: float = -5.0
        self._costmap_res:      float =  0.1
        self._costmap_cols:     int   =  100
        self._costmap_rows:     int   =  100

        # ── waypoint list ─────────────────────────────────────────────────────
        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx:    int = 0

        # ── goal publishing ───────────────────────────────────────────────────
        self._last_published_goal: tuple | None = None

        # ── QoS ──────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Bool,              '/poacher_visible', self._vis_cb,      10)
        self.create_subscription(Odometry,          '/odom',            self._drone_cb,    sensor_qos)
        self.create_subscription(Odometry,          '/poacher_odom',    self._poacher_cb,  sensor_qos)
        self.create_subscription(OccupancyGrid,     '/costmap',         self._costmap_cb,  reliable_qos)
        self.create_subscription(Float32MultiArray, '/processed_scan',  self._scan_cb,     sensor_qos)
        self.create_subscription(Float64MultiArray, '/scan_metadata',   self._meta_cb,     10)

        # ── publishers ───────────────────────────────────────────────────────
        self._goal_pub   = self.create_publisher(Point,        '/global_goal',    reliable_qos)
        self._marker_pub = self.create_publisher(Marker,       '/mission_marker', 10)
        self._state_pub  = self.create_publisher(String,       '/mission_state',  10)
        self._pdf_pub    = self.create_publisher(OccupancyGrid,'/pdf_map',        reliable_qos)

        self.create_timer(1.0 / goal_hz, self._tick)
        self.create_timer(5.0,           self._republish_goal)
        self.create_timer(0.5,           self._publish_pdf_map)

        self._clear_markers()
        self.get_logger().info('MissionNode ready')

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _vis_cb(self, msg: Bool) -> None:
        self._vis_buffer.append(msg.data)
        if len(self._vis_buffer) > self._VIS_DEBOUNCE_N:
            self._vis_buffer.pop(0)
        if len(self._vis_buffer) < self._VIS_DEBOUNCE_N:
            return
        if not all(v == self._vis_buffer[0] for v in self._vis_buffer):
            return

        confirmed = self._vis_buffer[0]

        if confirmed and not self._visible:
            self._visible = True
            self.get_logger().info('Poacher confirmed VISIBLE')

        elif not confirmed and self._visible:
            self._visible = False
            if self._poacher_received:
                self._lkp_x    = self._poacher_x
                self._lkp_y    = self._poacher_y
                self._lkp_time = time.monotonic()
            self.get_logger().info(
                f'Poacher confirmed LOST at LKP ({self._lkp_x:.2f}, {self._lkp_y:.2f})')
            self._reset_search()

        if self._visible and self._poacher_received:
            self._lkp_x = self._poacher_x
            self._lkp_y = self._poacher_y

    def _drone_cb(self, msg: Odometry) -> None:
        self._drone_x = msg.pose.pose.position.x
        self._drone_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._drone_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

    def _poacher_cb(self, msg: Odometry) -> None:
        self._poacher_x = msg.pose.pose.position.x + self._poacher_offset_x
        self._poacher_y = msg.pose.pose.position.y + self._poacher_offset_y
        if not self._poacher_received:
            self._lkp_x    = self._poacher_x
            self._lkp_y    = self._poacher_y
            self._lkp_time = time.monotonic()
            self._mission_state = STATE_GOTO
            self._last_published_goal = None
            self.get_logger().info(
                f'LKP seeded: ({self._lkp_x:.2f}, {self._lkp_y:.2f})')
        self._poacher_received = True

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self._costmap_origin_x = msg.info.origin.position.x
        self._costmap_origin_y = msg.info.origin.position.y
        self._costmap_res      = msg.info.resolution
        self._costmap_cols     = msg.info.width
        self._costmap_rows     = msg.info.height
        self._costmap_data = np.array(
            msg.data, dtype=np.float32
        ).reshape(self._costmap_rows, self._costmap_cols)

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        self._ranges    = list(msg.data)
        self._scan_ready = bool(self._ranges)

    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 3:
            self._angle_min = msg.data[0]
            self._angle_inc = msg.data[2]

    # ─────────────────────────────────────────────────────────────────────────
    # Search reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_search(self) -> None:
        self._mission_state       = STATE_GOTO
        self._in_search           = False
        self._pdf                 = None
        self._pdf_cleared         = None
        self._pdf_goal            = None
        self._waypoints           = []
        self._wp_idx              = 0
        self._last_published_goal = None

    # ─────────────────────────────────────────────────────────────────────────
    # PDF operations
    # ─────────────────────────────────────────────────────────────────────────

    def _init_pdf(self) -> None:
        half = self._radius
        self._pdf_origin_x = self._lkp_x - half
        self._pdf_origin_y = self._lkp_y - half
        n = max(4, int(2.0 * half / self._res))
        self._pdf_rows = n
        self._pdf_cols = n

        xs = self._pdf_origin_x + (np.arange(n) + 0.5) * self._res
        ys = self._pdf_origin_y + (np.arange(n) + 0.5) * self._res
        XX, YY = np.meshgrid(xs, ys)
        sigma = 0.1   # near-point at t=0 — diffusion spreads it over time
        dist2 = (XX - self._lkp_x)**2 + (YY - self._lkp_y)**2
        pdf   = np.exp(-dist2 / (2.0 * sigma**2))
        self._pdf         = pdf / pdf.sum()
        self._pdf_cleared = np.zeros((n, n), dtype=bool)  # nothing cleared yet
        self._pdf_goal    = None
        self._last_diffuse_t = time.monotonic()
        self.get_logger().info('PDF initialised')

    def _diffuse_pdf(self) -> None:
        """
        Incrementally diffuse using only dt since last call.
        D = v²  where v = 0.22 m/s → D = 0.004 m²/s (poacher max speed).
        After diffusion, reapply the permanent cleared mask so confirmed-empty
        cells never regain probability regardless of diffusion.
        """
        if self._pdf is None or self._pdf_cleared is None:
            return

        now = time.monotonic()
        dt  = now - self._last_diffuse_t
        self._last_diffuse_t = now

        if dt <= 0.0:
            return

        sigma_cells = max(0.3, math.sqrt(2.0 * self._D * dt) / self._res)
        ksize       = max(3, int(6 * sigma_cells) | 1)
        half        = ksize // 2
        ax          = np.arange(-half, half + 1)
        k1d         = np.exp(-ax**2 / (2.0 * sigma_cells**2))
        k1d        /= k1d.sum()
        kernel      = np.outer(k1d, k1d)
        diffused    = convolve(self._pdf, kernel, mode='reflect')

        # Reapply cleared mask — diffusion must not refill confirmed-empty cells
        diffused[self._pdf_cleared] = 0.0

        total = diffused.sum()
        if total > 1e-9:
            self._pdf = diffused / total
        else:
            self._pdf = diffused

    def _erase_scan_sector(self) -> None:
        """
        Zero PDF cells confirmed empty by the current scan up to
        scan_clear_range. Writes to the permanent cleared mask so
        diffusion cannot refill these cells in future ticks.
        """
        if self._pdf is None or self._pdf_cleared is None or not self._scan_ready:
            return
        changed = False
        for i, r in enumerate(self._ranges):
            if r <= 0.0:
                continue
            beam_angle = self._drone_yaw + self._angle_min + i * self._angle_inc
            steps = int(min(r, self._scan_clear_range) / self._res)
            for s in range(1, steps + 1):
                wx = self._drone_x + s * self._res * math.cos(beam_angle)
                wy = self._drone_y + s * self._res * math.sin(beam_angle)
                col = int((wx - self._pdf_origin_x) / self._res)
                row = int((wy - self._pdf_origin_y) / self._res)
                if 0 <= row < self._pdf_rows and 0 <= col < self._pdf_cols:
                    if not self._pdf_cleared[row, col]:
                        self._pdf_cleared[row, col] = True
                        self._pdf[row, col] = 0.0
                        changed = True
        if changed:
            total = self._pdf.sum()
            if total > 1e-9:
                self._pdf /= total

    def _pdf_max(self) -> float:
        if self._pdf is None:
            return 0.0
        return float(self._pdf.max())

    def _pdf_exhausted(self) -> bool:
        """
        PDF is exhausted when fewer than 1% of cells have any probability
        mass — meaning scan erasure has confirmed almost everywhere empty.
        Uses cell count rather than raw max value, which depends on grid size.
        """
        if self._pdf is None:
            return True
        nonzero = np.count_nonzero(self._pdf > 1e-9)
        total   = self._pdf_rows * self._pdf_cols
        return nonzero < max(1, int(0.01 * total))

    def _best_pdf_goal(self) -> tuple | None:
        """Highest-probability in-bounds cell."""
        if self._pdf is None:
            return None
        masked = self._pdf.copy()
        for row in range(self._pdf_rows):
            wy = self._pdf_origin_y + (row + 0.5) * self._res
            for col in range(self._pdf_cols):
                wx = self._pdf_origin_x + (col + 0.5) * self._res
                if not (ARENA_X_MIN <= wx <= ARENA_X_MAX
                        and ARENA_Y_MIN <= wy <= ARENA_Y_MAX):
                    masked[row, col] = 0.0
        if masked.max() <= 0.0:
            return None
        idx      = np.argmax(masked)
        row, col = divmod(int(idx), self._pdf_cols)
        return (self._pdf_origin_x + (col + 0.5) * self._res,
                self._pdf_origin_y + (row + 0.5) * self._res)

    # ─────────────────────────────────────────────────────────────────────────
    # Waypoint generation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_waypoints(self) -> None:
        if self._costmap_data is None:
            self.get_logger().warn('No costmap — cannot build waypoints')
            return

        obs_mask  = self._costmap_data >= COST_OBS_MIN
        unvisited = self._costmap_data == 0.0
        fringe    = binary_dilation(obs_mask) & unvisited & ~obs_mask

        if not np.any(fringe):
            self.get_logger().warn('No fringe cells found')
            return

        rows, cols = np.where(fringe)
        wx = self._costmap_origin_x + (cols + 0.5) * self._costmap_res
        wy = self._costmap_origin_y + (rows + 0.5) * self._costmap_res

        in_bounds = ((wx >= ARENA_X_MIN) & (wx <= ARENA_X_MAX) &
                     (wy >= ARENA_Y_MIN) & (wy <= ARENA_Y_MAX))
        wx, wy = wx[in_bounds], wy[in_bounds]

        if len(wx) == 0:
            self.get_logger().warn('No in-bounds fringe cells')
            return

        dists = np.hypot(wx - self._lkp_x, wy - self._lkp_y)
        order = np.argsort(dists)
        wx, wy = wx[order], wy[order]

        kept: list[tuple] = []
        for x, y in zip(wx.tolist(), wy.tolist()):
            if not any(math.hypot(x - kx, y - ky) < self._arr_r
                       for kx, ky in kept):
                kept.append((x, y))

        self._waypoints = kept
        self._wp_idx    = 0
        self.get_logger().info(
            f'Waypoint list ({len(kept)} points):')
        for i, (x, y) in enumerate(kept):
            self.get_logger().info(
                f'  [{i}]  ({x:.2f}, {y:.2f})  '
                f'dist={math.hypot(x-self._lkp_x, y-self._lkp_y):.2f} m')

    # ─────────────────────────────────────────────────────────────────────────
    # Main tick
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self._poacher_received:
            self._publish_state(STATE_INIT)
            return

        if self._visible:
            # ── TRACK ────────────────────────────────────────────────────────
            self._mission_state = STATE_TRACK
            goal = (self._poacher_x, self._poacher_y)

        else:
            # ── GOTO LKP ─────────────────────────────────────────────────────
            if not self._in_search:
                dist_lkp = math.hypot(
                    self._drone_x - self._lkp_x,
                    self._drone_y - self._lkp_y)
                if dist_lkp > self._arr_r:
                    self._mission_state = STATE_GOTO
                    goal = (self._lkp_x, self._lkp_y)
                else:
                    self._in_search = True
                    self._init_pdf()
                    self.get_logger().info('Arrived at LKP — starting PDF search')

            if self._in_search:
                # ── Phase A: PDF search ───────────────────────────────────────
                if self._mission_state != STATE_WAYPOINT:
                    self._mission_state = STATE_PDF
                    self._diffuse_pdf()
                    self._erase_scan_sector()

                    if self._pdf_exhausted():
                        # PDF exhausted — switch to waypoint phase
                        self.get_logger().info('PDF exhausted — building waypoint list')
                        self._build_waypoints()
                        self._mission_state = STATE_WAYPOINT
                        self._pdf_goal = None

                    else:
                        # Pick new PDF goal only when needed
                        if self._pdf_goal is None:
                            self._pdf_goal = self._best_pdf_goal()
                        else:
                            dist = math.hypot(
                                self._drone_x - self._pdf_goal[0],
                                self._drone_y - self._pdf_goal[1])
                            if dist < self._arr_r:
                                self.get_logger().info(
                                    f'PDF goal reached — picking next')
                                self._pdf_goal = self._best_pdf_goal()

                        goal = self._pdf_goal or (self._lkp_x, self._lkp_y)

                # ── Phase B: Waypoint search ──────────────────────────────────
                if self._mission_state == STATE_WAYPOINT:
                    if not self._waypoints:
                        self._build_waypoints()
                        if not self._waypoints:
                            goal = (self._lkp_x, self._lkp_y)
                            self._publish_goal(goal)
                            self._publish_marker(goal)
                            self._publish_state(self._mission_state)
                            return

                    cx, cy = self._waypoints[self._wp_idx]
                    if math.hypot(self._drone_x - cx, self._drone_y - cy) < self._arr_r:
                        self._wp_idx += 1
                        self.get_logger().info(
                            f'Waypoint {self._wp_idx-1} reached — '
                            f'moving to {self._wp_idx}/{len(self._waypoints)}')
                        if self._wp_idx >= len(self._waypoints):
                            self.get_logger().info('All waypoints visited — rebuilding')
                            self._waypoints = []
                            self._wp_idx    = 0
                            goal = (self._lkp_x, self._lkp_y)
                            self._publish_goal(goal)
                            self._publish_marker(goal)
                            self._publish_state(self._mission_state)
                            return

                    goal = self._waypoints[self._wp_idx]

        self._publish_goal(goal)
        self._publish_marker(goal)
        self._publish_state(self._mission_state)

    # ─────────────────────────────────────────────────────────────────────────
    # Publishers
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_goal(self, goal: tuple) -> None:
        if goal == self._last_published_goal:
            return
        self._last_published_goal = goal
        pt = Point()
        pt.x, pt.y, pt.z = goal[0], goal[1], 0.0
        self._goal_pub.publish(pt)
        self.get_logger().info(f'New goal → ({goal[0]:.2f}, {goal[1]:.2f})')

    def _republish_goal(self) -> None:
        if self._last_published_goal is not None:
            pt = Point()
            pt.x, pt.y, pt.z = (self._last_published_goal[0],
                                 self._last_published_goal[1], 0.0)
            self._goal_pub.publish(pt)

    def _publish_state(self, state: str) -> None:
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)

    def _publish_pdf_map(self) -> None:
        if self._pdf is None or self._mission_state == STATE_TRACK:
            return
        grid = OccupancyGrid()
        grid.header.stamp       = self.get_clock().now().to_msg()
        grid.header.frame_id    = 'odom'
        grid.info.resolution    = self._res
        grid.info.width         = self._pdf_cols
        grid.info.height        = self._pdf_rows
        grid.info.origin.position.x = self._pdf_origin_x
        grid.info.origin.position.y = self._pdf_origin_y
        grid.info.origin.orientation.w = 1.0
        flat = self._pdf.flatten()
        mx   = float(flat.max())
        if mx > 0:
            scaled = (flat / mx * 99.0).clip(0, 99).astype(np.uint8)
        else:
            scaled = np.zeros(len(flat), dtype=np.uint8)
        grid.data = [int(v) for v in scaled]
        self._pdf_pub.publish(grid)

    def _publish_marker(self, goal: tuple) -> None:
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.ns, m.id        = 'mission', 0
        m.type            = Marker.ARROW
        m.action          = Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = 0.08, 0.16, 0.0

        if self._mission_state == STATE_TRACK:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.2, 0.9
        elif self._mission_state == STATE_GOTO:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.9
        elif self._mission_state == STATE_PDF:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.5, 1.0, 0.9
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 0.9

        def pt3(x, y):
            p = Point(); p.x, p.y, p.z = x, y, 0.2; return p

        m.points = [pt3(self._drone_x, self._drone_y), pt3(goal[0], goal[1])]
        self._marker_pub.publish(m)

        # Goal sphere
        s = Marker()
        s.header = m.header
        s.ns, s.id = 'mission', 1
        s.type   = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x, s.pose.position.y, s.pose.position.z = goal[0], goal[1], 0.2
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.3
        s.color = m.color
        self._marker_pub.publish(s)

        # Waypoint cubes (Phase B only)
        if self._mission_state == STATE_WAYPOINT:
            for i, (wx, wy) in enumerate(self._waypoints):
                w = Marker()
                w.header = m.header
                w.ns, w.id = 'waypoints', i
                w.type   = Marker.CUBE
                w.action = Marker.ADD
                w.pose.position.x, w.pose.position.y, w.pose.position.z = wx, wy, 0.1
                w.pose.orientation.w = 1.0
                w.scale.x = w.scale.y = w.scale.z = 0.15
                if i < self._wp_idx:
                    w.color.r, w.color.g, w.color.b, w.color.a = 0.5, 0.5, 0.5, 0.5
                elif i == self._wp_idx:
                    w.color.r, w.color.g, w.color.b, w.color.a = 0.0, 1.0, 1.0, 1.0
                else:
                    w.color.r, w.color.g, w.color.b, w.color.a = 1.0, 1.0, 1.0, 0.7
                self._marker_pub.publish(w)

    def _clear_markers(self) -> None:
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.action          = Marker.DELETEALL
        self._marker_pub.publish(m)


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