#!/usr/bin/env python3
"""
poacher_tf_relay.py

The original burger SDF publishes TF on the global /tf topic with plain
frame names (odom->base_footprint). When two burgers run simultaneously,
both publish to /tf using the same frame names which causes conflicts.

This node:
  1. Reads /poacher/odom (which correctly has the poacher's pose)
  2. Publishes the odom->base_footprint transform on /tf with namespaced
     frame names: poacher/odom -> poacher/base_footprint
  3. Also publishes a static poacher/odom->odom identity transform so
     Nav2 can bridge between the two frames if needed.

This completely bypasses the DiffDrive TF output and generates clean,
namespaced TF from the odometry topic instead.
"""

#!/usr/bin/env python3
"""
poacher_tf_relay.py

Generates namespaced TF for the poacher robot from /poacher/odom, and
anchors poacher/odom into the global odom frame via a static transform
so both robots share the same world coordinate system.

The offset between frames = poacher_spawn - main_robot_spawn.
Default: main robot at (-2.0, -0.5), poacher at (2.0, 0.0)
-> offset = (4.0, 0.5)

Override with ROS parameters:
  poacher_x, poacher_y  – poacher spawn position
  drone_x,   drone_y    – main robot spawn position
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


class PoacherTFRelay(Node):

    def __init__(self):
        super().__init__('poacher_tf_relay')

        # Spawn positions – must match your launch arguments
        self.declare_parameter('poacher_x', 2.0)
        self.declare_parameter('poacher_y', 0.0)
        self.declare_parameter('drone_x',  -2.0)
        self.declare_parameter('drone_y',  -0.5)

        px = self.get_parameter('poacher_x').value
        py = self.get_parameter('poacher_y').value
        dx = self.get_parameter('drone_x').value
        dy = self.get_parameter('drone_y').value

        # Offset: how far poacher/odom origin is from odom origin
        self._offset_x = px - dx
        self._offset_y = py - dy

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._tf_pub        = self.create_publisher(TFMessage, '/tf',        100)
        self._tf_static_pub = self.create_publisher(TFMessage, '/tf_static', static_qos)

        self.create_subscription(
            Odometry, '/poacher/odom', self._odom_cb, sensor_qos
        )

        self._publish_static()
        self.create_timer(5.0, self._publish_static)

        self.get_logger().info(
            f'PoacherTFRelay ready  |  '
            f'odom->poacher/odom offset=({self._offset_x:.2f}, {self._offset_y:.2f})'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        # Dynamic: poacher/odom -> poacher/base_footprint
        t = TransformStamped()
        t.header.stamp    = msg.header.stamp
        t.header.frame_id = 'poacher/odom'
        t.child_frame_id  = 'poacher/base_footprint'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation
        tf_msg = TFMessage()
        tf_msg.transforms.append(t)
        self._tf_pub.publish(tf_msg)

    def _publish_static(self) -> None:
        now = self.get_clock().now().to_msg()
        static_msg = TFMessage()
        static_msg.transforms = [
            # Anchor poacher/odom into the global odom frame
            self._static_tf(now, 'odom', 'poacher/odom',
                            self._offset_x, self._offset_y, 0.0),
            # Sensor chain
            self._static_tf(now, 'poacher/base_footprint', 'poacher/base_link',
                            0.0, 0.0, 0.010),
            self._static_tf(now, 'poacher/base_link', 'poacher/base_scan',
                            -0.032, 0.0, 0.171),
            self._static_tf(now, 'poacher/base_scan',
                            'poacher/base_scan/hls_lfcd_lds',
                            0.0, 0.0, 0.0),
        ]
        self._tf_static_pub.publish(static_msg)

    def _static_tf(self, stamp, parent, child, x, y, z):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = parent
        t.child_frame_id  = child
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w    = 1.0
        return t


def main(args=None):
    rclpy.init(args=args)
    node = PoacherTFRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


def main(args=None):
    rclpy.init(args=args)
    node = PoacherTFRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
