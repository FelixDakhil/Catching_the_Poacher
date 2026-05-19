import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

class SenseNode(Node):
    def __init__(self):
        super().__init__('sense_node')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishes [front, left, right] distances
        self.dist_pub = self.create_publisher(Float32MultiArray, '/distances', 10)

        # Publishes 36 sector densities for the histogram
        self.hist_pub = self.create_publisher(Float32MultiArray, '/histogram', 10)

    def scan_callback(self, msg):
        ranges = list(msg.ranges)

        def valid_min(s):
            valid = [r for r in s if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        # Distances
        dist = Float32MultiArray()
        dist.data = [
            valid_min(ranges[0:10] + ranges[350:]),
            valid_min(ranges[80:100]),
            valid_min(ranges[260:280])
        ]
        self.dist_pub.publish(dist)

        # Histogram — 36 sectors of 10 degrees each
        sectors = []
        for i in range(36):
            chunk = ranges[i*10:(i*10)+10]
            valid = [r for r in chunk if msg.range_min < r < msg.range_max]
            min_r = min(valid) if valid else msg.range_max
            density = max(0.0, 1.0 - min_r / msg.range_max)
            sectors.append(round(density, 3))

        hist = Float32MultiArray()
        hist.data = sectors
        self.hist_pub.publish(hist)

def main():
    rclpy.init()
    rclpy.spin(SenseNode())

if __name__ == '__main__':
    main()