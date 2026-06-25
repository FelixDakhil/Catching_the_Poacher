#!/usr/bin/env python3
"""
global_planner_node.py  -  D* Lite global planner for TurtleBot3.

Stack
-----
  cost_map.py  -> /costmap  -> GlobalPlannerNode -> /goal_point -> plan_node (VFH+)
                                                  -> /global_path -> RViz

Usage
-----
  python3 global_planner_node.py
  ros2 topic pub --times 5 /global_goal geometry_msgs/msg/Point "{x: 3.0, y: 2.0, z: 0.0}"

Output chain (existing stack unchanged)
  GlobalPlannerNode -> /goal_point (Point)
  plan_node (VFH+)  -> /vfh_command (Float64MultiArray)
  act_node          -> /cmd_vel (TwistStamped)
"""

import heapq
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WAYPOINT_RADIUS  = 0.30
OBSTACLE_THRESH  = 50        # occupancy value (0-100) treated as impassable
PUBLISH_RATE_S   = 0.5
WAYPOINT_SPACING = 8
REPLAN_EVERY_N   = 10        # replan every Nth costmap update (costmap = 5 Hz)
INF = float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# D* Lite  –  clean implementation using lazy-deletion heap
# ─────────────────────────────────────────────────────────────────────────────
#
# Key insight: D* Lite searches BACKWARDS (goal → start).
# rhs(s) is always current; g(s) may lag behind.
# The heap uses lazy deletion: we never remove stale entries,
# we just ignore them when popped (their tag won't match the counter).
#
# Every time we push a cell we give it a new unique tag so we can
# identify and discard the old copies cheaply.

class DStarLite:

    def __init__(self) -> None:
        self._rows  = 0
        self._cols  = 0
        self._cost: np.ndarray | None = None

        # g and rhs stored as plain dicts; missing key = INF
        self._g:   dict[tuple, float] = {}
        self._rhs: dict[tuple, float] = {}

        # Lazy-deletion heap: entries are (k1, k2, tag, row, col)
        # tag = self._push_count[cell] at push time;
        # popped entry is stale if stored tag != current tag
        self._heap:       list = []
        self._push_tag:   dict[tuple, int] = {}  # cell -> latest push tag
        self._push_count: int = 0

        self._start: tuple | None = None
        self._goal:  tuple | None = None
        self._km:    float = 0.0
        self._last_start: tuple | None = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def set_map(self, cost_grid: np.ndarray) -> None:
        """Load cost grid and reset all search state."""
        self._rows, self._cols = cost_grid.shape
        self._cost = cost_grid.copy().astype(np.float32)
        self._full_reset()

    def _full_reset(self) -> None:
        self._g.clear()
        self._rhs.clear()
        self._heap.clear()
        self._push_tag.clear()
        self._push_count = 0
        self._km = 0.0
        self._last_start = self._start
        # Re-seed goal
        if self._goal is not None:
            self._rhs[self._goal] = 0.0
            self._push(self._goal)

    def set_start(self, row: int, col: int) -> None:
        """Update robot position. Must be called before set_goal."""
        new = (row, col)
        if new != self._start:
            if self._start is not None:
                self._km += self._h(self._last_start, self._start)
            self._last_start = self._start
            self._start = new

    def set_goal(self, row: int, col: int) -> None:
        """Set goal and reset search. Must be called after set_start."""
        self._goal = (row, col)
        self._full_reset()

    def update_cells(self, changes: list) -> None:
        if self._cost is None:
            return
        for r, c, new_cost in changes:
            if self._in_bounds(r, c):
                self._cost[r, c] = float(new_cost)
                self._update_vertex((r, c))
                for nb in self._neighbours((r, c)):
                    self._update_vertex(nb)

    # ── public plan ───────────────────────────────────────────────────────────

    def plan(self) -> list:
        """Run D* Lite. Returns [(row,col),...] start→goal or []."""
        if self._start is None or self._goal is None or self._cost is None:
            return []

        self._compute()

        g_s = self._g.get(self._start, INF)
        g_g = self._g.get(self._goal,  INF)

        if g_s >= 1e8:
            return []

        # Greedy trace start → goal following minimum cost+g
        path    = [self._start]
        seen    = {self._start}
        current = self._start

        for _ in range(self._rows * self._cols):
            if current == self._goal:
                break
            nbs = self._neighbours(current)
            if not nbs:
                return []

            best     = None
            best_val = INF
            for nb in nbs:
                v = self._step_cost(current, nb) + self._g.get(nb, INF)
                if v < best_val:
                    best_val, best = v, nb

            if best is None or best in seen or best_val >= 1e8:
                return []

            path.append(best)
            seen.add(best)
            current = best

        return path if current == self._goal else []

    # ── cell cost lookup ──────────────────────────────────────────────────────

    def cell_cost(self, row: int, col: int) -> float:
        """Public accessor for the raw cost at a cell (used by goal nudging)."""
        if self._cost is None or not self._in_bounds(row, col):
            return INF
        return float(self._cost[row, col])

    # ── internals ─────────────────────────────────────────────────────────────

    def _h(self, a, b) -> float:
        """Octile distance heuristic (admissible for 8-connected grid)."""
        if a is None or b is None:
            return 0.0
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        return max(dr, dc) + (math.sqrt(2)-1) * min(dr, dc)

    def _key(self, s: tuple) -> tuple:
        g   = self._g.get(s,   INF)
        rhs = self._rhs.get(s, INF)
        mn  = min(g, rhs)
        k1  = mn + self._h(self._start, s) + self._km
        return (k1, mn)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _neighbours(self, s: tuple) -> list:
        r, c = s
        return [
            (r+dr, c+dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and self._in_bounds(r+dr, c+dc)
        ]

    def _step_cost(self, a: tuple, b: tuple) -> float:
        """
        Traversal cost of entering cell b from a.
        Free cell = 1.0, partial cost cell scales up, obstacle = very large.
        """
        cell = float(self._cost[b[0], b[1]])
        if cell >= 1e8:
            return 1e9
        diag = math.sqrt(2) if abs(a[0]-b[0])==1 and abs(a[1]-b[1])==1 else 1.0
        # +1 ensures every traversable step has strictly positive cost
        return diag * (1.0 + cell / 100.0)

    # ── lazy-deletion heap ────────────────────────────────────────────────────

    def _push(self, s: tuple) -> None:
        """Push s with its current key. Old copies become stale via tag."""
        self._push_count += 1
        tag = self._push_count
        self._push_tag[s] = tag
        k = self._key(s)
        heapq.heappush(self._heap, (k[0], k[1], tag, s[0], s[1]))

    def _top_key(self) -> tuple:
        """Return the key of the cheapest VALID heap entry."""
        while self._heap:
            k1, k2, tag, r, c = self._heap[0]
            s = (r, c)
            if self._push_tag.get(s) == tag:
                return (k1, k2)
            heapq.heappop(self._heap)   # discard stale
        return (INF, INF)

    def _pop(self):
        """Pop and return the cheapest valid (key, cell). Discards stale entries."""
        while self._heap:
            k1, k2, tag, r, c = heapq.heappop(self._heap)
            s = (r, c)
            if self._push_tag.get(s) == tag:
                # Valid entry — clear its tag so it's no longer "in heap"
                del self._push_tag[s]
                return (k1, k2), s
            # Stale — discard and continue
        return None, None

    def _update_vertex(self, s: tuple) -> None:
        """Recompute rhs(s). If s is inconsistent, push it onto the heap."""
        if s != self._goal:
            nbs = self._neighbours(s)
            self._rhs[s] = min(
                (self._step_cost(s, nb) + self._g.get(nb, INF) for nb in nbs),
                default=INF
            )
        if self._g.get(s, INF) != self._rhs.get(s, INF):
            self._push(s)

    def _compute(self) -> None:
        """
        Expand inconsistent cells until start is consistent and
        no queued cell has a better key than start.
        This is the verbatim D* Lite main loop.
        """
        iters = 0
        limit = self._rows * self._cols * 4

        while iters < limit:
            iters += 1
            top_k   = self._top_key()
            start_k = self._key(self._start)

            g_s   = self._g.get(self._start,   INF)
            rhs_s = self._rhs.get(self._start, INF)

            # Termination: start is locally consistent AND queue has nothing better
            if top_k >= start_k and g_s == rhs_s:
                break

            k, s = self._pop()
            if s is None:
                break   # heap empty

            new_k = self._key(s)

            if k < new_k:
                # Key outdated — re-insert with correct key
                self._push(s)

            elif self._g.get(s, INF) > self._rhs.get(s, INF):
                # Over-consistent: set g = rhs and propagate to predecessors
                self._g[s] = self._rhs[s]
                for nb in self._neighbours(s):
                    self._update_vertex(nb)

            else:
                # Under-consistent: raise g to INF, re-queue, propagate
                self._g[s] = INF
                self._update_vertex(s)
                for nb in self._neighbours(s):
                    self._update_vertex(nb)


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class GlobalPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('global_planner_node')

        self._dstar = DStarLite()

        # Map state
        self._map_res:       float | None = None
        self._map_origin_x:  float = 0.0
        self._map_origin_y:  float = 0.0
        self._map_cols:      int   = 0
        self._map_rows:      int   = 0
        self._map_loaded:    bool  = False
        self._costmap_count: int   = 0

        # Robot state
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0

        # Goal state
        self._goal_x:     float | None = None
        self._goal_y:     float | None = None
        self._goal_active: bool = False

        # Waypoints
        self._waypoints: list = []
        self._wp_index:  int  = 0

        # Thread safety
        self._planning_now: bool           = False
        self._plan_lock:    threading.Lock = threading.Lock()

        # QoS
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(Point,         '/global_goal', self._goal_cb,    reliable_qos)
        self.create_subscription(Odometry,      '/odom',        self._odom_cb,    sensor_qos)
        self.create_subscription(OccupancyGrid, '/costmap',     self._costmap_cb, reliable_qos)

        # Publishers
        self._goal_pub = self.create_publisher(Point, '/goal_point',  10)
        self._path_pub = self.create_publisher(Path,  '/global_path', 10)

        self.create_timer(PUBLISH_RATE_S, self._publish_timer_cb)

        self.get_logger().info(
            '  ros2 topic pub --once /global_goal '
            'geometry_msgs/msg/Point "{x: 3.0, y: 2.0, z: 0.0}"'
        )

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _world_to_grid(self, wx: float, wy: float) -> tuple | None:
        if self._map_res is None:
            return None
        col = int((wx - self._map_origin_x) / self._map_res)
        row = int((wy - self._map_origin_y) / self._map_res)
        if 0 <= col < self._map_cols and 0 <= row < self._map_rows:
            return (row, col)
        return None

    def _grid_to_world(self, row: int, col: int) -> tuple:
        wx = self._map_origin_x + (col + 0.5) * self._map_res
        wy = self._map_origin_y + (row + 0.5) * self._map_res
        return wx, wy

    # ── goal nudging ──────────────────────────────────────────────────────────

    def _nudge_goal_cell(
        self,
        start_cell: tuple,
        goal_cell: tuple,
        goal_weight: float = 1.0,
        robot_weight: float = 1.0,
        max_radius: int = 20,
    ) -> tuple | None:
        """
        If goal_cell is blocked, search outward in expanding square rings
        for a free cell, scoring each candidate by a WEIGHTED combination
        of its distance to the goal and its distance to the robot:

            score = goal_weight * dist_to_goal + robot_weight * dist_to_robot

        Lower score wins. With robot_weight > 0, candidates on the robot's
        side of an obstacle are preferred over geometrically-closer cells
        that happen to sit on the far side of a wall — without needing a
        full line-of-sight raycast, since the weighting alone biases the
        ring search toward the robot.

        Tuning:
          goal_weight  high  -> stick close to the original goal position
          robot_weight high  -> prefer cells nearer the robot (safer, but
                                 may end up further from the intended goal)
          Equal weights (default) balance the two.

        Returns the original cell unchanged if it's already free.
        Returns None if nothing free is found within max_radius cells.
        """
        gr, gc = goal_cell
        sr, sc = start_cell

        if self._dstar.cell_cost(gr, gc) < 1e8:
            return goal_cell   # already free — no nudge needed

        for radius in range(1, max_radius + 1):
            candidates = []

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    # Only the ring boundary — interior already checked
                    # at smaller radii
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    r, c = gr + dr, gc + dc
                    if not self._dstar._in_bounds(r, c):
                        continue
                    if self._dstar.cell_cost(r, c) >= 1e8:
                        continue   # still blocked

                    dist_to_goal  = math.hypot(r - gr, c - gc)
                    dist_to_robot = math.hypot(r - sr, c - sc)
                    score = goal_weight * dist_to_goal + robot_weight * dist_to_robot

                    candidates.append((score, r, c))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                _, r, c = candidates[0]
                return (r, c)

        return None   # nothing free within max_radius

    # ── /odom ─────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

        if not self._goal_active or not self._waypoints:
            return
        if self._wp_index >= len(self._waypoints):
            return

        wx, wy = self._waypoints[self._wp_index]
        if math.hypot(self._robot_x - wx, self._robot_y - wy) < WAYPOINT_RADIUS:
            self._wp_index += 1
            if self._wp_index >= len(self._waypoints):
                self.get_logger().info(
                    f'Global goal reached ({self._goal_x:.2f}, {self._goal_y:.2f})'
                )
                self._goal_active = False
                self._waypoints   = []
            else:
                nwx, nwy = self._waypoints[self._wp_index]
                self.get_logger().info(
                    f'Waypoint {self._wp_index+1}/{len(self._waypoints)} '
                    f'({nwx:.2f}, {nwy:.2f})'
                )

    # ── /global_goal ──────────────────────────────────────────────────────────

    def _goal_cb(self, msg: Point) -> None:
        self._goal_x      = msg.x
        self._goal_y      = msg.y
        self._goal_active = True
        self._wp_index    = 0
        self.get_logger().info(f'Goal received: ({msg.x:.2f}, {msg.y:.2f})')

        if not self._map_loaded:
            self.get_logger().warn('No costmap')
            return

        self._spawn_replan()

    # ── /costmap ──────────────────────────────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self._map_res      = msg.info.resolution
        self._map_origin_x = msg.info.origin.position.x
        self._map_origin_y = msg.info.origin.position.y
        self._map_cols     = msg.info.width
        self._map_rows     = msg.info.height

        raw = np.array(msg.data, dtype=np.int8).reshape(
            (self._map_rows, self._map_cols))

        # Cells >= OBSTACLE_THRESH are impassable; rest get proportional cost
        cost_grid = np.where(
            raw >= OBSTACLE_THRESH,
            1e9,
            raw.astype(np.float32)
        ).astype(np.float32)

        if not self._map_loaded:
            self._dstar.set_map(cost_grid)
            self._map_loaded = True
            if self._goal_active:
                self._spawn_replan()
            return

        self._costmap_count += 1
        if self._costmap_count % REPLAN_EVERY_N != 0:
            return

        self._dstar.set_map(cost_grid)
        if self._goal_active and not self._planning_now:
            self._spawn_replan()

    # ── planning ──────────────────────────────────────────────────────────────

    def _spawn_replan(self) -> None:
        if self._planning_now:
            return
        self._planning_now = True
        threading.Thread(target=self._do_replan, daemon=True).start()

    def _do_replan(self) -> None:
        try:
            self._replan()
        finally:
            self._planning_now = False

    def _replan(self) -> None:
        start_cell = self._world_to_grid(self._robot_x, self._robot_y)
        goal_cell  = self._world_to_grid(self._goal_x,  self._goal_y)

        if start_cell is None:
            self.get_logger().warn(
                f'Robot ({self._robot_x:.2f}, {self._robot_y:.2f}) outside map'
            )
            return

        if goal_cell is None:
            self.get_logger().warn(
                f'Goal ({self._goal_x:.2f}, {self._goal_y:.2f}) outside map  '
                f'x:[{self._map_origin_x:.1f} -> '
                f'{self._map_origin_x + self._map_cols*self._map_res:.1f}]  '
                f'y:[{self._map_origin_y:.1f} -> '
                f'{self._map_origin_y + self._map_rows*self._map_res:.1f}]'
            )
            return

        # ── nudge the goal to the nearest free cell if it's blocked ──────────
        sc = self._dstar.cell_cost(*start_cell)
        gc = self._dstar.cell_cost(*goal_cell)

        if gc >= 1e8:
            nudged = self._nudge_goal_cell(start_cell, goal_cell)
            if nudged is None:
                self.get_logger().warn(
                    f'Goal cell {goal_cell} is blocked and no free cell exists '
                    f'on the line of sight from the robot — goal unreachable.'
                )
                return

            nwx, nwy = self._grid_to_world(*nudged)
            self.get_logger().info(
                f'Goal cell {goal_cell} was blocked (cost={gc:.0f}) — '
                f'nudged to {nudged} = ({nwx:.2f}, {nwy:.2f})'
            )
            goal_cell = nudged
            gc = self._dstar.cell_cost(*goal_cell)

        self.get_logger().info(
            f'Replanning  start: {start_cell} ({sc:.0f})  '
            f'goal: {goal_cell} ({gc:.0f})'
        )

        with self._plan_lock:
            self._dstar.set_start(start_cell[0], start_cell[1])
            self._dstar.set_goal(goal_cell[0],   goal_cell[1])
            path_cells = self._dstar.plan()

        if not path_cells:
            self.get_logger().warn('D* Lite: no path found')
            return

        self._waypoints = self._cells_to_waypoints(path_cells)
        self._wp_index  = 0
        self.get_logger().info(
            f'Path found: {len(path_cells)} cells -> '
            f'{len(self._waypoints)} waypoints  '
            f'first=({self._waypoints[0][0]:.2f}, {self._waypoints[0][1]:.2f})'
        )
        self._publish_path(path_cells)

    def _cells_to_waypoints(self, path_cells: list) -> list:
        wps = []
        for i, (r, c) in enumerate(path_cells):
            if i % WAYPOINT_SPACING == 0 or i == len(path_cells) - 1:
                wps.append(self._grid_to_world(r, c))
        return wps

    # ── publish timer ─────────────────────────────────────────────────────────

    def _publish_timer_cb(self) -> None:
        if not self._goal_active or not self._waypoints:
            return
        if self._wp_index >= len(self._waypoints):
            return
        wx, wy = self._waypoints[self._wp_index]
        pt = Point()
        pt.x = wx
        pt.y = wy
        pt.z = 0.0
        self._goal_pub.publish(pt)

    # ── RViz path ─────────────────────────────────────────────────────────────

    def _publish_path(self, path_cells: list) -> None:
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for r, c in path_cells:
            wx, wy = self._grid_to_world(r, c)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x  = wx
            ps.pose.position.y  = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()