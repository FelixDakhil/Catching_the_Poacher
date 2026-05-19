import rclpy
import robot_path
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

def f0(time):

    cmd = TwistStamped()
        # TwistStamped requires a header with a timestamp
        cmd.header.stamp = self.get_clock().now().to_msg()
    
    if time < 3.0:
            cmd.twist.angular.z = 0.15 
            cmd.twist.linear.x = 0.15
            self.get_logger().info('Turning')
        else:
            cmd.twist.angular.z = 0.0 
            cmd.twist.linear.x = 0.0

            if self.running:
                self.get_logger().info(f'Moved after {elapsed:.1f}s')
                self.running = False

        self.cmd_pub.publish(cmd)