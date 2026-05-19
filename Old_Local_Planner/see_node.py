import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

# See Node

class SeeNode(Node):
    def __init__(self):
        super().__init__('see_node')

        
        self._pos_x: float = 0.0
        self._pos_y: float = 0.0
        self._yaw:   float = 0.0
        self._ranges: np.ndarray = np.full(360, np.nan)

        
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 10)
        self.create_subscription(Odometry,  '/odom', self._cb_odom, 10)

        
        self._pub = self.create_publisher(Float32MultiArray, '/sensor_data', 10)
        self.create_timer(0.1, self._publish)

        self.get_logger().info('See No Evil')

    # ------------------------------------------------------------------
    def _cb_odom(self, msg: Odometry):
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = float(np.arctan2(siny, cosy))

    def _cb_scan(self, msg: LaserScan):
        """
        TurtleBot3 burger: 360 rays, angle_min=-pi, angle_increment=pi/180.
        Clamp to [range_min, range_max]; mark the rest NaN.
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[ranges < msg.range_min] = np.nan
        ranges[ranges > msg.range_max] = np.nan
        # Resize defensively — burger is always 360 but be safe
        if len(ranges) != 360:
            indices = np.round(
                np.linspace(0, len(ranges) - 1, 360)).astype(int)
            ranges = ranges[indices]
        self._ranges = ranges

    def _publish(self):
        out = Float32MultiArray()
        out.data = (
            [self._pos_x, self._pos_y, self._yaw]
            + self._ranges.tolist()
        )
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SeeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
