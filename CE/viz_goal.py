#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point

class RvizGoalRelay(Node):
    def __init__(self):
        super().__init__('rviz_goal_relay')
        self.create_subscription(PoseStamped, '/goal_pose', self._cb, 10)
        self._pub = self.create_publisher(Point, '/global_goal', 10)
        self.get_logger().info('Its 2D Goal Pose in RViz2')

    def _cb(self, msg: PoseStamped):
        pt = Point()
        pt.x = msg.pose.position.x
        pt.y = msg.pose.position.y
        pt.z = 0.0
        self._pub.publish(pt)
        self.get_logger().info(f'Goal → ({pt.x:.2f}, {pt.y:.2f})')

def main():
    rclpy.init()
    rclpy.spin(RvizGoalRelay())

if __name__ == '__main__':
    main()