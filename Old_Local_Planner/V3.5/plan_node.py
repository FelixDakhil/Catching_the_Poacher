import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

SECTORS     = 36
THRESHOLD   = 0.4   # density above this = blocked
SECTOR_DEG  = 10    # degrees per sector

class PlanNode(Node):
    def __init__(self):
        super().__init__('plan_node')
        self.create_subscription(Float32MultiArray, '/histogram', self.plan_callback, 10)
        self.action_pub = self.create_publisher(String, '/action', 10)

    def plan_callback(self, msg):
        densities = list(msg.data)

        # --- MASK --- mark each sector blocked (1) or free (0)
        mask = [1 if d > THRESHOLD else 0 for d in densities]

        self.get_logger().info(self.render_mask(mask))

        # --- PLAN --- find best valley to drive towards
        action = self.find_valley(mask)

        out = String()
        out.data = action
        self.action_pub.publish(out)

    def find_valley(self, mask):
        # Sector 0 = front, sector 9 = left, sector 27 = right
        front_free = mask[0] == 0

        if front_free:
            return 'forward'

        # Count free sectors on each side
        left_free  = sum(1 for i in range(1, 18)  if mask[i] == 0)
        right_free = sum(1 for i in range(18, 36) if mask[i] == 0)

        if left_free == 0 and right_free == 0:
            return 'reverse'       # completely surrounded
        elif right_free > left_free:
            return 'turn_right'
        else:
            return 'turn_left'

    def render_mask(self, mask):
        # Prints a simple ASCII ring in the terminal showing blocked sectors
        ring = ''
        for i, m in enumerate(mask):
            if i == 0:
                ring += '[F]'      # mark front
            else:
                ring += '█' if m else '░'
        return f'mask: {ring}'

def main():
    rclpy.init()
    rclpy.spin(PlanNode())

if __name__ == '__main__':
    main()