import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped

class ActNode(Node):
    def __init__(self):
        super().__init__('act_node')
        self.create_subscription(String, '/action', self.act_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    def act_callback(self, msg):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        if msg.data == 'forward':
            cmd.twist.linear.x  =  0.15
        elif msg.data == 'turn_left':
            cmd.twist.angular.z =  0.5
        elif msg.data == 'turn_right':
            cmd.twist.angular.z = -0.5
        elif msg.data == 'reverse':
            cmd.twist.linear.x  = -0.15

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Acting: {msg.data}')

def main():
    rclpy.init()
    rclpy.spin(ActNode())

if __name__ == '__main__':
    main()