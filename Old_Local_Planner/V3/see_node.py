#!/usr/bin/env python3
"""
see_node.py  –  Perception layer for TurtleBot3 in Gazebo.

Responsibilities
----------------
* Subscribe to /scan  (sensor_msgs/LaserScan)
* Clean / validate the raw readings (inf → max_range, NaN → max_range)
* Publish a ProcessedScan message on /processed_scan
  (std_msgs/Float32MultiArray with layout metadata)

Topic layout published on /processed_scan
------------------------------------------
  layout.dim[0].label  = "ranges"
  layout.dim[0].size   = number of beams
  layout.dim[0].stride = number of beams
  data                 = cleaned range values  [float32 ×N]

A second topic, /scan_metadata, carries angle_min / angle_max /
angle_increment / range_max as a Float64MultiArray so the planner
can reconstruct beam directions without re-subscribing to /scan.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float64MultiArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCAN_TOPIC        = "/scan"
PROCESSED_TOPIC   = "/processed_scan"
METADATA_TOPIC    = "/scan_metadata"

DEFAULT_MAX_RANGE = 3.5   # TurtleBot3 LDS-01 spec (metres)


class SeeNode(Node):
    """Perception node: cleans LaserScan and republishes for the planner."""

    def __init__(self) -> None:
        super().__init__("see_node")

        # ----- parameters --------------------------------------------------
        self.declare_parameter("max_range",     DEFAULT_MAX_RANGE)
        self.declare_parameter("min_range",     0.12)   # LDS-01 blind zone
        self.declare_parameter("publish_rate",  10.0)   # Hz (unused – event-driven)

        self._max_range = self.get_parameter("max_range").value
        self._min_range = self.get_parameter("min_range").value

        # ----- QoS matching Gazebo sensor best-effort ----------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----- subscriber --------------------------------------------------
        self._scan_sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._scan_callback,
            sensor_qos,
        )

        # ----- publishers --------------------------------------------------
        self._processed_pub = self.create_publisher(
            Float32MultiArray,
            PROCESSED_TOPIC,
            10,
        )
        self._meta_pub = self.create_publisher(
            Float64MultiArray,
            METADATA_TOPIC,
            10,
        )

        self.get_logger().info(
            f"SeeNode ready  |  listening on {SCAN_TOPIC}  "
            f"|  max_range={self._max_range} m"
        )

    # -----------------------------------------------------------------------
    # Callback
    # -----------------------------------------------------------------------
    def _scan_callback(self, msg: LaserScan) -> None:
        """Process one LaserScan frame and publish cleaned data."""
        max_r = msg.range_max if msg.range_max > 0.0 else self._max_range
        min_r = msg.range_min if msg.range_min > 0.0 else self._min_range

        cleaned: list[float] = []
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < min_r or r > max_r:
                cleaned.append(float(max_r))
            else:
                cleaned.append(float(r))

        n = len(cleaned)

        # ---- processed scan -----------------------------------------------
        proc_msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = "ranges"
        dim.size   = n
        dim.stride = n
        proc_msg.layout.dim.append(dim)
        proc_msg.layout.data_offset = 0
        proc_msg.data = cleaned
        self._processed_pub.publish(proc_msg)

        # ---- metadata (published every scan so planner always has it) -----
        meta_msg = Float64MultiArray()
        meta_msg.data = [
            float(msg.angle_min),
            float(msg.angle_max),
            float(msg.angle_increment),
            float(max_r),
        ]
        self._meta_pub.publish(meta_msg)

        self.get_logger().debug(
            f"Published {n} beams  "
            f"|  min={min(cleaned):.2f} m  max={max(cleaned):.2f} m"
        )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
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
