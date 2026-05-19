import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray

# The planning node

class PlanNode(Node):

    
    OBSTACLE_THRESHOLD = 0.5   # m
    INFLATION_DEG      = 10    
    NUM_SECTORS        = 72    
    GOAL_X             = 9.0
    GOAL_Y             = 0.5
    

    def __init__(self):
        super().__init__('plan_node')

        self.declare_parameter('goal_x', self.GOAL_X)
        self.declare_parameter('goal_y', self.GOAL_Y)
        self.declare_parameter('obstacle_threshold', self.OBSTACLE_THRESHOLD)

        self._goal_x    = self.get_parameter('goal_x').value
        self._goal_y    = self.get_parameter('goal_y').value
        self._threshold = self.get_parameter('obstacle_threshold').value

        self._sensor_data: list | None = None

        self.create_subscription(
            Float32MultiArray, '/sensor_data', self._cb_sensor, 10)
        self._pub = self.create_publisher(
            Float32MultiArray, '/plan_cmd', 10)

        self.create_timer(0.1, self._plan)
        self.get_logger().info(
            f'Plan no Evil — to ({self._goal_x}, {self._goal_y})')

    
    def _cb_sensor(self, msg: Float32MultiArray):
        self._sensor_data = list(msg.data)

    def _plan(self):
        if self._sensor_data is None:
            return

        pos_x  = self._sensor_data[0]
        pos_y  = self._sensor_data[1]
        yaw    = self._sensor_data[2]
        ranges = np.array(self._sensor_data[3:], dtype=np.float32)  

        
        dist_to_goal = np.hypot(self._goal_x - pos_x, self._goal_y - pos_y)
        if dist_to_goal < 0.3:
            self.get_logger().info('Goal reached!')
            self._publish(0.0, 0.0, stop=True)
            return

        hist = self._build_histogram(ranges)

        heading_offset, confidence = self._select_heading(
            hist, pos_x, pos_y, yaw)

        self._publish(heading_offset, confidence)

    def _build_histogram(self, ranges: np.ndarray) -> np.ndarray:
        
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
                hist[s] = 1.0 - (min_r / self._threshold)

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
       
        FREE_THRESHOLD = 0.3  


        dx = self._goal_x - pos_x
        dy = self._goal_y - pos_y
        goal_bearing_world = np.arctan2(dy, dx)
        goal_bearing_robot = goal_bearing_world - yaw


        goal_bearing_robot = goal_bearing_robot % (2 * np.pi)
        goal_sector = int(goal_bearing_robot / (2 * np.pi) * self.NUM_SECTORS) \
                      % self.NUM_SECTORS

        free_mask    = hist < FREE_THRESHOLD
        free_indices = np.where(free_mask)[0]

        if len(free_indices) == 0:
            
            return 0.0, 0.0

        diffs = np.abs(free_indices - goal_sector)
        diffs = np.minimum(diffs, self.NUM_SECTORS - diffs)
        best_sector = free_indices[int(np.argmin(diffs))]


        sector_angle = (best_sector / self.NUM_SECTORS) * 2 * np.pi
        heading_offset = sector_angle - goal_bearing_robot

        heading_offset = (heading_offset + np.pi) % (2 * np.pi) - np.pi


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
