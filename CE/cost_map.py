import math
import struct
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


MAP_WIDTH_M    = 10.0
MAP_HEIGHT_M   = 10.0
RESOLUTION     = 0.1     # metres per cell

COST_FREE      = 0.0
COST_MAX       = 100.0

HIT_INCREMENT  = 30.0    
MISS_DECREMENT = 5.0     
DECAY_RATE     = 0.02    

PUBLISH_RATE_S     = 0.2   # robot-facing costmap publish rate (5 Hz)
VIZ_PUBLISH_RATE_S = 2.0   # RViz2-facing costmap publish rate (0.5 Hz)
MAX_RANGE_M    = 3.5     # = DEFAULT_MAX_RANGE

MAP_FRAME   = 'map'
ODOM_FRAME  = 'odom'
ROBOT_FRAME = 'base_link'  

# Stolen from Petr Viktorin GITHUB
def bresenham_line(x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy;  x0 += sx
        if e2 <  dx:
            err += dx;  y0 += sy

# Formatting from https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html 
def make_pointcloud2(header: Header, points: list) -> PointCloud2:
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    data = bytearray()
    for (x, y, z) in points:
        data += struct.pack('fff', float(x), float(y), float(z))
    msg = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = len(points)
    msg.fields       = fields
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(points)
    msg.data         = bytes(data)
    msg.is_dense     = True
    return msg


# The actual node
class DynamicCostmapNode(Node):

    def __init__(self):
        super().__init__('dynamic_costmap_node')

        self.cols     = int(MAP_WIDTH_M  / RESOLUTION)
        self.rows     = int(MAP_HEIGHT_M / RESOLUTION)
        self.origin_x = -MAP_WIDTH_M  / 2.0  # -50 on both
        self.origin_y = -MAP_HEIGHT_M / 2.0 
        self.costmap  = np.zeros((self.rows, self.cols), dtype=np.float32)

        self._hit_points: list = []

        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._odom_tf: TransformStamped | None = None

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # QoS for the RViz2 publisher — RELIABLE with depth 1 matches
        # RViz2's default subscriber QoS. The slow 1 Hz publish rate is
        # what prevents the queue from filling up, not the QoS policy.
        viz_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(LaserScan, '/scan', self.scan_callback, sensor_qos)
        self.create_subscription(Odometry,  '/odom', self.odom_callback, sensor_qos)

        # /costmap      — RELIABLE, 5 Hz — for global_planner_node and other nodes
        self.map_pub     = self.create_publisher(OccupancyGrid, '/costmap',     10)

        # /costmap/viz  — BEST_EFFORT, 1 Hz — for RViz2 only
        self.map_viz_pub = self.create_publisher(OccupancyGrid, '/costmap/viz', viz_qos)

        self.pose_pub  = self.create_publisher(PoseStamped,   '/drone_pose', 10)
        self.hits_pub  = self.create_publisher(PointCloud2,   '/scan_hits',  10)

        # Two separate timers: fast for the robot stack, slow for RViz2
        self.create_timer(PUBLISH_RATE_S,     self.publish_all)
        self.create_timer(VIZ_PUBLISH_RATE_S, self._publish_costmap_viz)

        self.get_logger().info(
            f'DynamicCostmapNode ready | '
            f'grid={self.cols}x{self.rows} | res={RESOLUTION}m | '
            f'frames: {MAP_FRAME} -> {ODOM_FRAME} -> {ROBOT_FRAME} | '
            f'costmap -> /costmap ({1/PUBLISH_RATE_S:.0f} Hz) '
            f'+ /costmap/viz ({1/VIZ_PUBLISH_RATE_S:.0f} Hz for RViz2)'
        )

    def world_to_grid(self, wx, wy):
        col = int((wx - self.origin_x) / RESOLUTION)
        row = int((wy - self.origin_y) / RESOLUTION)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return col, row
        return None

    def odom_callback(self, msg):

        stamp = msg.header.stamp

        # map to odom
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp    = stamp
        map_to_odom.header.frame_id = MAP_FRAME
        map_to_odom.child_frame_id  = ODOM_FRAME
        map_to_odom.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(map_to_odom)

        # odom to base_link
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp    = stamp
        odom_to_base.header.frame_id = ODOM_FRAME
        odom_to_base.child_frame_id  = ROBOT_FRAME
        odom_to_base.transform.translation.x = msg.pose.pose.position.x
        odom_to_base.transform.translation.y = msg.pose.pose.position.y
        odom_to_base.transform.translation.z = msg.pose.pose.position.z
        odom_to_base.transform.rotation      = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(odom_to_base)

        self._odom_tf = odom_to_base

    def scan_callback(self, msg: LaserScan):

        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                ROBOT_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup {MAP_FRAME}->{ROBOT_FRAME} failed: {e}'
            )
            return

        robot_wx = tf.transform.translation.x
        robot_wy = tf.transform.translation.y

        robot_cell = self.world_to_grid(robot_wx, robot_wy)
        if robot_cell is None:
            self.get_logger().warn(
                f'Drone ({robot_wx:.1f}, {robot_wy:.1f}) outside map bounds.'
            )
            return

        robot_col, robot_row = robot_cell

        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        robot_yaw = 2.0 * math.atan2(qz, qw)

        angle     = msg.angle_min + robot_yaw
        angle_inc = msg.angle_increment
        hit_pts   = []

        for r in msg.ranges:
            angle += angle_inc

            if math.isnan(r) or math.isinf(r):
                continue

            r_used = min(r, MAX_RANGE_M)
            is_hit = (r < msg.range_max) and (r < MAX_RANGE_M)

            end_wx   = robot_wx + r_used * math.cos(angle)
            end_wy   = robot_wy + r_used * math.sin(angle)
            end_cell = self.world_to_grid(end_wx, end_wy)

            if end_cell is None:
                continue

            end_col, end_row = end_cell

            for (col, row) in bresenham_line(robot_col, robot_row, end_col, end_row):
                if 0 <= col < self.cols and 0 <= row < self.rows:
                    self.costmap[row, col] = max(
                        COST_FREE,
                        self.costmap[row, col] - MISS_DECREMENT
                    )

            if is_hit:
                self.costmap[end_row, end_col] = min(
                    COST_MAX,
                    self.costmap[end_row, end_col] + HIT_INCREMENT
                )
                hit_pts.append((end_wx, end_wy, 0.0))

        self._hit_points = hit_pts

        # time decay
        self.costmap *= (1.0 - DECAY_RATE)

    def publish_all(self):
        """Fast timer (5 Hz) — publishes everything the robot stack needs."""
        stamp = self.get_clock().now().to_msg()
        self._publish_costmap(stamp)
        self._publish_drone_pose(stamp)
        self._publish_scan_hits(stamp)

    def _build_grid_msg(self, data: np.ndarray, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp              = stamp
        msg.header.frame_id           = MAP_FRAME
        msg.info.resolution           = RESOLUTION
        msg.info.width                = self.cols
        msg.info.height               = self.rows
        msg.info.origin.position.x    = self.origin_x
        msg.info.origin.position.y    = self.origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data.flatten().astype(np.int8).tolist()
        return msg

    def _publish_costmap(self, stamp):
        """Publish to /costmap (RELIABLE, 5 Hz) for the robot stack."""
        self.map_pub.publish(self._build_grid_msg(self.costmap, stamp))

    def _publish_costmap_viz(self):
        """
        Publish to /costmap/viz for RViz2.

        Uses a zero timestamp (ros::Time(0)) instead of the current clock.
        This bypasses RViz2's TF-aware message filter entirely — the filter
        only applies to messages with non-zero stamps because it needs to
        look up a transform at that exact time. With stamp=0 RViz2 renders
        the map immediately without waiting for a matching TF entry, which
        eliminates the 'queue is full' and 'timestamp earlier than TF cache'
        warnings.

        Point RViz2's Map display at /costmap/viz, not /costmap.
        """
        from builtin_interfaces.msg import Time as RosTime
        zero_stamp = RosTime()   # sec=0, nanosec=0
        self.map_viz_pub.publish(self._build_grid_msg(self.costmap, zero_stamp))

    def _publish_drone_pose(self, stamp):
        if self._odom_tf is None:
            return
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = MAP_FRAME
        msg.pose.position.x = self._odom_tf.transform.translation.x
        msg.pose.position.y = self._odom_tf.transform.translation.y
        msg.pose.position.z = 0.0
        msg.pose.orientation = self._odom_tf.transform.rotation
        self.pose_pub.publish(msg)

    def _publish_scan_hits(self, stamp):
        if not self._hit_points:
            return
        header = Header()
        header.stamp    = stamp
        header.frame_id = MAP_FRAME
        self.hits_pub.publish(make_pointcloud2(header, self._hit_points))


#-----

def main(args=None):
    rclpy.init(args=args)
    node = DynamicCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()