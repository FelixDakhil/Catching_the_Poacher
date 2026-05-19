# drone_spa_node.py
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class DroneNavNode(Node):
    def __init__(self):
        super().__init__('drone_spa_node')

        # --- parameters ---
        self.declare_parameter('forward_speed', 0.8)
        self.declare_parameter('max_angular', 1.2)
        self.declare_parameter('obstacle_threshold', 1.5)   # metres
        self.declare_parameter('safety_radius', 0.5)        # metres
        self.declare_parameter('vfh_sectors', 72)           # 5° each

        self.forward_speed     = self.get_parameter('forward_speed').value
        self.max_angular       = self.get_parameter('max_angular').value
        self.obstacle_threshold= self.get_parameter('obstacle_threshold').value
        self.safety_radius     = self.get_parameter('safety_radius').value
        self.num_sectors       = self.get_parameter('vfh_sectors').value

        # --- state ---
        self.scan: LaserScan | None = None
        self.yaw   = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

        # Goal in world frame (set externally or hard-coded)
        self.goal_x = 10.0
        self.goal_y = 0.0

        # --- ROS2 interfaces ---
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._cb_scan, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Main loop at 20 Hz
        self.create_timer(0.05, self._loop)

    # ------------------------------------------------------------------ #
    #  SEE                                                                 #
    # ------------------------------------------------------------------ #
    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _cb_odom(self, msg: Odometry):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = np.arctan2(siny, cosy)

    # ------------------------------------------------------------------ #
    #  PLAN  — VFH+ polar obstacle histogram                              #
    # ------------------------------------------------------------------ #
    def _build_histogram(self) -> np.ndarray:
        """
        Build a 1-D binary histogram over self.num_sectors sectors.
        A sector is 1 (blocked) if any lidar return within that angular
        slice is closer than obstacle_threshold.
        """
        scan = self.scan
        hist = np.zeros(self.num_sectors, dtype=float)

        num_rays = len(scan.ranges)
        sector_size = num_rays // self.num_sectors

        for s in range(self.num_sectors):
            start = s * sector_size
            end   = start + sector_size
            rays  = scan.ranges[start:end]
            # Filter inf/nan
            valid = [r for r in rays if scan.range_min < r < scan.range_max]
            if valid and min(valid) < self.obstacle_threshold:
                # Weight by proximity: closer = higher certainty
                hist[s] = 1.0 - (min(valid) / self.obstacle_threshold)

        return hist

    def _vfh_select_heading(self, hist: np.ndarray) -> float | None:
        """
        Find the valley (free sector group) whose centre bearing is
        closest to the goal bearing. Returns a heading offset in radians,
        or None if completely blocked.
        """
        # Bearing to goal in robot frame
        dx = self.goal_x - self.pos_x
        dy = self.goal_y - self.pos_y
        goal_bearing = np.arctan2(dy, dx) - self.yaw  # relative to robot
        goal_sector  = int((goal_bearing % (2 * np.pi))
                           / (2 * np.pi) * self.num_sectors) % self.num_sectors

        threshold = 0.5  # binary-ish: sectors above this are "blocked"
        free = hist < threshold

        if not np.any(free):
            return None  # completely stuck — caller should handle

        # Find all free sectors, pick the one closest to goal bearing
        free_indices = np.where(free)[0]
        # Angular distance (circular) from each free sector to goal sector
        diffs = np.abs(free_indices - goal_sector)
        diffs = np.minimum(diffs, self.num_sectors - diffs)
        best_sector = free_indices[np.argmin(diffs)]

        # Convert sector index back to heading offset
        sector_angle = (best_sector / self.num_sectors) * 2 * np.pi
        heading_offset = sector_angle - np.pi  # centre at 0
        return float(np.clip(heading_offset, -np.pi, np.pi))

    # ------------------------------------------------------------------ #
    #  ACT                                                                 #
    # ------------------------------------------------------------------ #
    def _act(self, heading_offset: float | None):
        cmd = Twist()

        if heading_offset is None:
            # Completely blocked: rotate in place to find a gap
            cmd.linear.x  = 0.0
            cmd.angular.z = self.max_angular * 0.5
            self.pub_cmd.publish(cmd)
            return

        # Forward bias: always maintain minimum forward speed
        # Scale linear speed down when turning sharply
        turn_factor   = 1.0 - abs(heading_offset) / np.pi
        cmd.linear.x  = max(0.15, self.forward_speed * turn_factor)

        # P-controller on heading error
        cmd.angular.z = float(np.clip(
            1.5 * heading_offset, -self.max_angular, self.max_angular))

        self.pub_cmd.publish(cmd)

    # ------------------------------------------------------------------ #
    #  MAIN LOOP                                                           #
    # ------------------------------------------------------------------ #
    def _loop(self):
        if self.scan is None:
            return  # waiting for first scan

        # Check if goal reached
        dist = np.hypot(self.goal_x - self.pos_x, self.goal_y - self.pos_y)
        if dist < 0.5:
            self.pub_cmd.publish(Twist())  # stop
            self.get_logger().info('Goal reached!')
            return

        hist           = self._build_histogram()   # SEE
        heading_offset = self._vfh_select_heading(hist)  # PLAN
        self._act(heading_offset)                  # ACT


def main():
    rclpy.init()
    node = DroneNavNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()