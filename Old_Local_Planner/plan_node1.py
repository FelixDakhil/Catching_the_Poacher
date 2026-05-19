#!/usr/bin/env python3
"""
PLAN node — TurtleBot3 Burger / turtlebot3_world
Subscribes to /sensor_data (from SEE node).
Runs a VFH+ polar histogram to pick the safest heading toward the goal.
Publishes a 2-element Float32MultiArray on /plan_cmd:
  [0]  target_heading  (radians, in robot frame, 0 = straight ahead)
  [1]  confidence      (0.0 = blocked, 1.0 = clear path to goal)
Goal modes
----------
1. Fixed goal   — publish once to /goal           (geometry_msgs/PointStamped)
2. Moving target — publish continuously to /goal  (same topic, same message)

Both use the same topic. If messages keep arriving, the robot keeps
chasing the latest one. If they stop, the robot heads for the last
known position.

Send a goal from the terminal:
  ros2 topic pub --once /goal geometry_msgs/PointStamped \
    "{header: {frame_id: 'odom'}, point: {x: 2.0, y: 1.5, z: 0.0}}"

Update it while running (moving target, 2 Hz):
  ros2 topic pub -r 2 /goal geometry_msgs/PointStamped \
    "{header: {frame_id: 'odom'}, point: {x: 3.0, y: -1.0, z: 0.0}}"
"""

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped


class PlanNode(Node):

    # ---- tuneable parameters ----------------------------------------
    OBSTACLE_THRESHOLD = 0.8   # metres — react earlier; pillars need more space
    INFLATION_DEG      = 20    # degrees to inflate — wider margin around pillars
    FREE_THRESHOLD     = 0.1   # sector score below this = passable
                               # lower = stricter, avoids near-obstacle sectors
    NUM_SECTORS        = 72    # 5 deg per sector
    GOAL_X             = 0.0   # default — overridden by /goal topic
    GOAL_Y             = 0.0
    GOAL_REACHED_DIST  = 0.3   # metres
    # -----------------------------------------------------------------

    def __init__(self):
        super().__init__('plan_node')

        self.declare_parameter('goal_x',             self.GOAL_X)
        self.declare_parameter('goal_y',             self.GOAL_Y)
        self.declare_parameter('obstacle_threshold', self.OBSTACLE_THRESHOLD)
        self.declare_parameter('inflation_deg',      float(self.INFLATION_DEG))
        self.declare_parameter('free_threshold',     self.FREE_THRESHOLD)

        self._goal_x        = self.get_parameter('goal_x').value
        self._goal_y        = self.get_parameter('goal_y').value
        self._threshold     = self.get_parameter('obstacle_threshold').value
        self._inflation_deg = self.get_parameter('inflation_deg').value
        self._free_thresh   = self.get_parameter('free_threshold').value
        self._goal_set  = False   # stays False until a /goal message arrives

        self._sensor_data: list | None = None

        self.create_subscription(
            Float32MultiArray, '/sensor_data', self._cb_sensor, 10)

        # /goal accepts a PointStamped — works for both fixed and moving targets
        self.create_subscription(
            PointStamped, '/goal', self._cb_goal, 10)

        self._pub = self.create_publisher(
            Float32MultiArray, '/plan_cmd', 10)

        self.create_timer(0.1, self._plan)
        self.get_logger().info(
            'PLAN node ready — waiting for goal on /goal topic.\n'
            '  Send one with:\n'
            '  ros2 topic pub --once /goal geometry_msgs/PointStamped '
            '"{header: {frame_id: \'odom\'}, point: {x: 2.0, y: 0.0, z: 0.0}}"')

    # ------------------------------------------------------------------
    def _cb_sensor(self, msg: Float32MultiArray):
        self._sensor_data = list(msg.data)

    def _cb_goal(self, msg: PointStamped):
        """Accept a new goal at any time — supports both fixed and moving targets."""
        self._goal_x = msg.point.x
        self._goal_y = msg.point.y
        if not self._goal_set:
            self.get_logger().info(
                f'Goal received: ({self._goal_x:.2f}, {self._goal_y:.2f}) '
                f'[frame: {msg.header.frame_id}]')
            self._goal_set = True
        else:
            self.get_logger().debug(
                f'Goal updated: ({self._goal_x:.2f}, {self._goal_y:.2f})')

    def _plan(self):
        if self._sensor_data is None:
            return

        if not self._goal_set:
            self.get_logger().info(
                'Waiting for goal — publish to /goal to start.',
                throttle_duration_sec=3.0)
            return

        pos_x  = self._sensor_data[0]
        pos_y  = self._sensor_data[1]
        yaw    = self._sensor_data[2]
        ranges = np.array(self._sensor_data[3:], dtype=np.float32)  # 360 values

        # ---- check goal reached ----
        dist_to_goal = np.hypot(self._goal_x - pos_x, self._goal_y - pos_y)
        if dist_to_goal < self.GOAL_REACHED_DIST:
            self.get_logger().info('Goal reached!')
            self._publish(0.0, 0.0, stop=True)
            return

        # ---- build VFH histogram ----
        hist = self._build_histogram(ranges)

        # ---- pick best sector ----
        heading_offset, confidence = self._select_heading(
            hist, pos_x, pos_y, yaw)

        self._publish(heading_offset, confidence)

    # ------------------------------------------------------------------
    def _build_histogram(self, ranges: np.ndarray) -> np.ndarray:
        """
        Map 360 lidar rays onto NUM_SECTORS sectors.
        Each sector value = obstacle density (0 = free, 1 = fully blocked).
        Sectors are then inflated by INFLATION_DEG to add safety margin.
        """
        rays_per_sector = 360 // self.NUM_SECTORS
        hist = np.zeros(self.NUM_SECTORS, dtype=float)

        for s in range(self.NUM_SECTORS):
            start = s * rays_per_sector
            end   = start + rays_per_sector
            chunk = ranges[start:end]
            valid = chunk[np.isfinite(chunk)]
            if len(valid) == 0:
                hist[s] = 1.0  # unknown = treat as blocked
                continue
            min_r = float(np.min(valid))
            if min_r < self._threshold:
                # Density weighted by proximity
                hist[s] = 1.0 - (min_r / self._threshold)

        # Inflate blocked sectors by INFLATION_DEG
        inf_sectors = max(1, self.INFLATION_DEG // (360 // self.NUM_SECTORS))
        kernel = np.ones(2 * inf_sectors + 1)
        inflated = np.convolve(
            np.tile(hist, 3), kernel, mode='same')[
                self.NUM_SECTORS: 2 * self.NUM_SECTORS]
        inflated = np.clip(inflated, 0.0, 1.0)

        return inflated

    def _select_heading(
        self,
        hist: np.ndarray,
        pos_x: float,
        pos_y: float,
        yaw: float,
    ) -> tuple[float, float]:
        """
        Find the free-sector valley whose centre is closest to the goal
        bearing. Returns (heading_offset_radians, confidence).
        heading_offset = 0 means straight ahead, positive = turn left.
        """
        FREE_THRESHOLD = 0.3  # sectors below this are considered passable

        # Goal bearing in robot frame
        dx = self._goal_x - pos_x
        dy = self._goal_y - pos_y
        goal_bearing_world = np.arctan2(dy, dx)
        goal_bearing_robot = goal_bearing_world - yaw
        # Normalise to [0, 2pi) then convert to sector index
        goal_bearing_robot = goal_bearing_robot % (2 * np.pi)
        goal_sector = int(goal_bearing_robot / (2 * np.pi) * self.NUM_SECTORS) \
                      % self.NUM_SECTORS

        free_mask    = hist < FREE_THRESHOLD
        free_indices = np.where(free_mask)[0]

        if len(free_indices) == 0:
            # Completely blocked — signal ACT to rotate
            return 0.0, 0.0

        # Circular distance from each free sector to the goal sector
        diffs = np.abs(free_indices - goal_sector)
        diffs = np.minimum(diffs, self.NUM_SECTORS - diffs)
        best_sector = free_indices[int(np.argmin(diffs))]

        # Convert best sector back to a heading offset in [-pi, pi]
        sector_angle = (best_sector / self.NUM_SECTORS) * 2 * np.pi
        heading_offset = sector_angle - goal_bearing_robot
        # Wrap to [-pi, pi]
        heading_offset = (heading_offset + np.pi) % (2 * np.pi) - np.pi

        # Confidence: 1 = best sector IS the goal sector
        min_diff = float(diffs[np.argmin(diffs)])
        confidence = max(0.0, 1.0 - min_diff / (self.NUM_SECTORS / 2))

        return float(heading_offset), float(confidence)

    def _publish(self, heading: float, confidence: float, stop: bool = False):
        msg = Float32MultiArray()
        msg.data = [-999.0, 0.0] if stop else [heading, confidence]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()