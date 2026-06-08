#!/usr/bin/env python3
"""
see_node.py  –  Perception layer for TurtleBot3 in Gazebo.

Responsibilities
----------------
* Subscribe to /scan  (sensor_msgs/LaserScan)
* Clean / validate the raw readings (inf → max_range, NaN → max_range)
* Publish cleaned ranges on /processed_scan  (Float32MultiArray)
* Publish beam geometry on /scan_metadata    (Float64MultiArray)

/scan_metadata data layout
--------------------------
  [angle_min, angle_max, angle_increment, range_max]
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float64MultiArray


SCAN_TOPIC        = "/scan"
PROCESSED_TOPIC   = "/processed_scan"
METADATA_TOPIC    = "/scan_metadata"
DEFAULT_MAX_RANGE = 3.5   # TurtleBot3 LDS-01 spec (metres)


class SeeNode(Node):
    """Perception node: cleans LaserScan and republishes for the planner."""

    def __init__(self) -> None:
        super().__init__("see_node")

        self.declare_parameter("max_range", DEFAULT_MAX_RANGE)
        self.declare_parameter("min_range", 0.1)   # LDS-01 blind zone

        self._max_range = self.get_parameter("max_range").value
        self._min_range = self.get_parameter("min_range").value

        # Match Gazebo sensor QoS (best-effort)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._scan_sub = self.create_subscription(
            LaserScan, SCAN_TOPIC, self._scan_callback, sensor_qos,
        )
        self._processed_pub = self.create_publisher(
            Float32MultiArray, PROCESSED_TOPIC, 10,
        )
        self._meta_pub = self.create_publisher(
            Float64MultiArray, METADATA_TOPIC, 10,
        )

        self.get_logger().info(
            f"SeeNode ready  |  {SCAN_TOPIC} → {PROCESSED_TOPIC}  "
            f"|  max_range={self._max_range} m"
        )

    # ------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan) -> None:
        max_r = msg.range_max if msg.range_max > 0.0 else self._max_range
        min_r = msg.range_min if msg.range_min > 0.0 else self._min_range

        cleaned: list[float] = []
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < min_r or r > max_r:
                cleaned.append(float(max_r))
            else:
                cleaned.append(float(r))

        n = len(cleaned)

        # Publish cleaned ranges
        proc = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = "ranges"
        dim.size   = n
        dim.stride = n
        proc.layout.dim.append(dim)
        proc.layout.data_offset = 0
        proc.data = cleaned
        self._processed_pub.publish(proc)

        # Publish beam geometry so plan_node can reconstruct directions
        meta = Float64MultiArray()
        meta.data = [
            float(msg.angle_min),
            float(msg.angle_max),
            float(msg.angle_increment),
            float(max_r),
        ]
        self._meta_pub.publish(meta)

        self.get_logger().debug(
            f"{n} beams  min={min(cleaned):.2f} m  max={max(cleaned):.2f} m"
        )


# ------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = SeeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
