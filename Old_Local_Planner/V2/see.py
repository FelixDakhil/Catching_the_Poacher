from std_msgs.msg import Float32MultiArray

class SenseNode(Node):
    def __init__(self):
        super().__init__('sense_node')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/distances', 10)

    def scan_callback(self, msg):
        ranges = msg.ranges

        def valid_min(s):
            valid = [r for r in s if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        out = Float32MultiArray()
        out.data = [
            valid_min(list(ranges[0:10]) + list(ranges[350:])),  # front
            valid_min(list(ranges[80:100])),                      # left
            valid_min(list(ranges[260:280]))                      # right
        ]
        self.pub.publish(out)

