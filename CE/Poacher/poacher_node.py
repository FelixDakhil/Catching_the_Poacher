#!/usr/bin/env python3
"""
poacher_node.py  –  Drives the poacher TurtleBot3 via Nav2 NavigateToPose.

Run after poacher_launch.py is up and Nav2 is active (~15 s after launch):
  python3 poacher_node.py --ros-args -p waypoints:="2.0,0.0; 4.0,1.0; 3.0,-2.0"

Waypoint format: semicolon-separated "x,y" pairs.
First waypoint = spawn position (skipped as a navigation target).
Waits for new /poacher_waypoints after reaching the last point.

Topics
------
  Sub:  /poacher/odom           nav_msgs/Odometry
  Sub:  /poacher_waypoints      geometry_msgs/PoseArray  runtime override
  Pub:  /poacher_odom           nav_msgs/Odometry        clean re-publish
  Pub:  /poacher_marker         visualization_msgs/Marker  red sphere
  Pub:  /poacher_path_marker    visualization_msgs/Marker  red path line strip
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker

COLOUR_SPHERE = (1.0, 0.0, 0.0, 1.0)
COLOUR_PATH   = (1.0, 0.35, 0.35, 0.85)


def _parse_waypoints(s: str) -> list[tuple[float, float]]:
    result = []
    for token in s.split(';'):
        parts = token.strip().split(',')
        if len(parts) == 2:
            try:
                result.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return result


class PoacherNode(Node):

    def __init__(self) -> None:
        super().__init__('poacher_node')

        self.declare_parameter('waypoints', '')
        self._waypoints = _parse_waypoints(
            self.get_parameter('waypoints').value
        )
        self._wp_index   = 1    # index 0 is spawn, head for index 1
        self._nav_active = False
        self._x = self._y = self._yaw = 0.0

        self._nav = ActionClient(
            self, NavigateToPose, '/poacher/navigate_to_pose'
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(
            Odometry, '/poacher/odom', self._odom_cb, sensor_qos
        )
        self.create_subscription(
            PoseArray, '/poacher_waypoints', self._wps_cb, 10
        )

        self._odom_pub   = self.create_publisher(Odometry, '/poacher_odom',        10)
        self._sphere_pub = self.create_publisher(Marker,   '/poacher_marker',      10)
        self._path_pub   = self.create_publisher(Marker,   '/poacher_path_marker', 10)

        self.create_timer(0.1, self._publish_markers)

        # Wait 10 s for Nav2 to be fully active before sending first goal
        self.create_timer(10.0, self._initial_goal)

        self.get_logger().info(
            f'PoacherNode ready  |  {len(self._waypoints)} waypoint(s)\n'
            '  Runtime override:\n'
            '  ros2 topic pub --times 1 /poacher_waypoints '
            'geometry_msgs/msg/PoseArray '
            '"{poses: [{position: {x: 2.0, y: 0.0}}, '
            '{position: {x: 4.0, y: 1.0}}]}"'
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        self._odom_pub.publish(msg)

    def _wps_cb(self, msg: PoseArray) -> None:
        wps = [(p.position.x, p.position.y) for p in msg.poses]
        if not wps:
            return
        self._waypoints  = wps
        self._wp_index   = 1
        self._nav_active = False
        self.get_logger().info(
            f'New waypoints: {len(wps)}  first={wps[0]}'
        )
        self._send_goal()

    def _initial_goal(self) -> None:
        if self._waypoints and self._wp_index < len(self._waypoints):
            self._send_goal()

    def _send_goal(self) -> None:
        if self._nav_active:
            return
        if not self._waypoints or self._wp_index >= len(self._waypoints):
            self.get_logger().info('All waypoints done – waiting for new queue.')
            return

        wx, wy = self._waypoints[self._wp_index]

        if not self._nav.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('Nav2 not ready – retry in 3 s')
            self.create_timer(3.0, self._send_goal)
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = wx
        goal.pose.pose.position.y = wy
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(
            f'Goal {self._wp_index}/{len(self._waypoints)-1}: '
            f'({wx:.2f}, {wy:.2f})'
        )
        self._nav_active = True
        fut = self._nav.send_goal_async(goal)
        fut.add_done_callback(self._accepted_cb)

    def _accepted_cb(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected – retry in 2 s')
            self._nav_active = False
            self.create_timer(2.0, self._send_goal)
            return
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future) -> None:
        self._nav_active = False
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Reached waypoint {self._wp_index}')
            self._wp_index += 1
            self._send_goal()
        else:
            self.get_logger().warn(f'Goal status {status} – retry in 2 s')
            self.create_timer(2.0, self._send_goal)

    def _publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()

        s = Marker()
        s.header.stamp = now
        s.header.frame_id = 'odom'
        s.ns, s.id = 'poacher', 0
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x = self._x
        s.pose.position.y = self._y
        s.pose.position.z = 0.15
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.3
        s.color.r, s.color.g, s.color.b, s.color.a = COLOUR_SPHERE
        self._sphere_pub.publish(s)

        p = Marker()
        p.header.stamp = now
        p.header.frame_id = 'odom'
        p.ns, p.id = 'poacher_path', 1
        p.type = Marker.LINE_STRIP
        p.action = Marker.ADD
        p.pose.orientation.w = 1.0
        p.scale.x = 0.05
        p.color.r, p.color.g, p.color.b, p.color.a = COLOUR_PATH

        def pt(x, y):
            q = Point(); q.x, q.y, q.z = x, y, 0.05; return q

        p.points.append(pt(self._x, self._y))
        for wx, wy in self._waypoints[self._wp_index:]:
            p.points.append(pt(wx, wy))
        p.action = Marker.DELETE if len(p.points) <= 1 else Marker.ADD
        self._path_pub.publish(p)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoacherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
