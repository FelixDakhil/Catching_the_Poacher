import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10) # With 10 being the queue size (its required)

        self.start_time = self.get_clock().now() #Syncs with Gazebo simulation time
        self.running = True

        # DO NOT USE while True
        frq = 0.1 # Frequency in Hz
        self.create_timer(frq, self.control_loop)
        self.get_logger().info('Timer starts')

    def control_loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        cmd = TwistStamped()
        # TwistStamped requires a header with a timestamp
        cmd.header.stamp = self.get_clock().now().to_msg()
        

        
        
        cmd.twist.angular.z = 0.0 
        cmd.twist.linear.x = 0.0

        if self.running:
            self.get_logger().info('Stopped')
            self.running = False

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = Robot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': # This is genuinely needed for the node to work in ROS2
    main()