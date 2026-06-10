#!/usr/bin/env python3
"""
poacher_detection_node.py  –  Detects whether the drone can "see" the poacher.

Visibility is determined by two conditions:
  1. RANGE  – the poacher is within the drone's LiDAR max range (3.5 m default)
  2. LINE OF SIGHT – the LiDAR beam in the direction of the poacher reads a
     range >= distance_to_poacher - tolerance.  If a wall is closer than the
     poacher in that direction, the poacher is occluded.

Publishes
---------
  /poacher_visible    std_msgs/Bool        True when poacher is visible
  /poacher_bearing    std_msgs/Float64     bearing to poacher in radians
                                           (robot frame, 0 = straight ahead)
  /detection_marker   visualization_msgs/Marker
                      Green line = visible, Red line = occluded

Subscribes
----------
  /odom               nav_msgs/Odometry    drone pose
  /poacher_odom       nav_msgs/Odometry    poacher pose (global frame)
  /processed_scan     std_msgs/Float32MultiArray  cleaned LiDAR from see_node
  /scan_metadata      std_msgs/Float64MultiArray  beam geometry from see_node

Parameters
----------
  los_tolerance   float  m  how much closer than the poacher a beam can
                            read before the poacher is considered occluded
                            (default 0.3 – accounts for poacher body size)
  max_range       float  m  max detection range (default 3.5)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64, Float32MultiArray, Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class PoacherDetectionNode(Node):

    def __init__(self) -> None:
        super().__init__('poacher_detection_node')

        self.declare_parameter('los_tolerance',   0.3)
        self.declare_parameter('max_range',       1.5)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.declare_parameter('poacher_spawn_x', 2.0)
        self.declare_parameter('poacher_spawn_y', 0.0)
        self.declare_parameter('drone_spawn_x',  -2.0)
        self.declare_parameter('drone_spawn_y',  -0.5)

        self._tol       = self.get_parameter('los_tolerance').value
        self._max_range = self.get_parameter('max_range').value   # used for ALL range checks

        # Offset to convert poacher local odom → drone world frame
        self._poacher_offset_x = (self.get_parameter('poacher_spawn_x').value
                                - self.get_parameter('drone_spawn_x').value)
        self._poacher_offset_y = (self.get_parameter('poacher_spawn_y').value
                                - self.get_parameter('drone_spawn_y').value)

        # Robot state
        self._drone_x:   float = 0.0
        self._drone_y:   float = 0.0
        self._drone_yaw: float = 0.0

        # Poacher state
        self._poacher_x:        float = 0.0
        self._poacher_y:        float = 0.0
        self._poacher_received: bool  = False

        # Scan state
        self._ranges:        list[float] = []
        self._angle_min:     float = 0.0
        self._angle_inc:     float = math.radians(1.0)
        self._meta_received: bool  = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Odometry,          '/odom',            self._drone_cb,   sensor_qos)
        self.create_subscription(Odometry,          '/poacher_odom',    self._poacher_cb, sensor_qos)
        self.create_subscription(Float32MultiArray, '/processed_scan',  self._scan_cb,    sensor_qos)
        self.create_subscription(Float64MultiArray, '/scan_metadata',   self._meta_cb,    10)

        self._vis_pub    = self.create_publisher(Bool,    '/poacher_visible',  10)
        self._bear_pub   = self.create_publisher(Float64, '/poacher_bearing',  10)
        self._marker_pub = self.create_publisher(Marker,  '/detection_marker', 10)

        self.create_timer(0.1, self._update)

        self.get_logger().info(
            f'PoacherDetectionNode ready  |  '
            f'max_range={self._max_range} m  los_tolerance={self._tol} m'
        )

    # -----------------------------------------------------------------------
    def _drone_cb(self, msg: Odometry) -> None:
        self._drone_x = msg.pose.pose.position.x
        self._drone_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._drone_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )

    def _poacher_cb(self, msg: Odometry) -> None:
        self._poacher_x = msg.pose.pose.position.x + self._poacher_offset_x
        self._poacher_y = msg.pose.pose.position.y + self._poacher_offset_y
        self._poacher_received = True

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        self._ranges = list(msg.data)

    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 3:
            self._angle_min     = msg.data[0]
            self._angle_inc     = msg.data[2]
            self._meta_received = True

    # -----------------------------------------------------------------------
    def _update(self) -> None:
        if not self._poacher_received or not self._meta_received or not self._ranges:
            return

        # Distance and world bearing to poacher
        dx   = self._poacher_x - self._drone_x
        dy   = self._poacher_y - self._drone_y
        dist = math.hypot(dx, dy)

        # Out of range
        if dist > self._max_range:
            self._publish(False, 0.0, dist)
            return

        # Bearing in robot frame
        world_bear  = math.atan2(dy, dx)
        robot_bear  = math.atan2(
            math.sin(world_bear - self._drone_yaw),
            math.cos(world_bear - self._drone_yaw),
        )

        # Find the LiDAR beam closest to that bearing
        beam_idx = int(round(
            (robot_bear - self._angle_min) / self._angle_inc
        )) % len(self._ranges)

        beam_range = self._ranges[beam_idx]

        # Clamp beam to max_range — keeps the LOS check consistent with the
        # range gate above. If the beam reads inf or beyond max_range, treat
        # it as max_range (open air up to the detection horizon).
        beam_range = min(beam_range, self._max_range)

        # Line-of-sight check: beam must reach at least as far as the poacher
        # (minus tolerance for the poacher's body size)
        visible = beam_range >= (dist - self._tol)

        self._publish(visible, robot_bear, dist)

    def _publish(self, visible: bool, bearing: float, dist: float) -> None:
        # Bool
        b = Bool()
        b.data = visible
        self._vis_pub.publish(b)

        # Bearing
        f = Float64()
        f.data = bearing
        self._bear_pub.publish(f)

        # Marker: line from drone to poacher, green=visible red=occluded
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.ns, m.id        = 'detection', 0
        m.type            = Marker.LINE_LIST
        m.action          = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.04

        if visible:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.9
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.5

        def pt(x, y):
            p = Point(); p.x, p.y, p.z = x, y, 0.15; return p

        m.points.append(pt(self._drone_x,   self._drone_y))
        m.points.append(pt(self._poacher_x, self._poacher_y))
        self._marker_pub.publish(m)

        self.get_logger().info(
            f'Poacher {"VISIBLE" if visible else "hidden":8s}  '
            f'dist={dist:.2f} m  '
            f'bearing={math.degrees(bearing):.1f}°',
            throttle_duration_sec=1.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoacherDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()