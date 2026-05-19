#!/usr/bin/env python3
"""
ACT node — TurtleBot3 Burger / turtlebot3_world
Subscribes to /plan_cmd (from PLAN node) and /scan (emergency stop).
Publishes geometry_msgs/TwistStamped on /cmd_vel.

TurtleBot3 burger limits:
  max linear  x: 0.22 m/s
  max angular z: 2.84 rad/s

Behaviour:
  - 1-second standstill on startup before any motion
  - Emergency stop + ERROR log when any ray < ESTOP_DISTANCE
  - Never command negative linear.x (always keep moving forward)
  - Scale linear speed down when turn is sharp
  - Minimum forward creep so robot doesn't stall mid-turn
  - If confidence == 0 (completely blocked), rotate in place to find gap
  - If heading sentinel == -999 (goal reached), publish zero and stop
"""

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class ActNode(Node):

    # TurtleBot3 burger hard limits
    MAX_LINEAR  = 0.20   # m/s  (slightly under 0.22 for safety)
    MAX_ANGULAR = 2.50   # rad/s

    # Tuning
    FORWARD_SPEED   = 0.15   # nominal cruise speed
    MIN_FORWARD     = 0.05   # minimum speed during sharp turns
    KP_ANGULAR      = 1.8    # proportional gain on heading error
    ROTATION_SPEED  = 0.8    # rad/s when completely blocked

    # Safety
    ESTOP_DISTANCE  = 0.1   # metres — hard stop if anything closer than this
    STARTUP_HOLD    = 1.0    # seconds of forced standstill on boot

    def __init__(self):
        super().__init__('act_node')

        self._heading       = 0.0
        self._confidence    = 0.0
        self._goal_reached  = False
        self._plan_received = False
        self._estopped      = False

        # Startup hold — record boot time, refuse motion for STARTUP_HOLD secs
        self._start_time = self.get_clock().now()
        self.get_logger().info(
            f'ACT node starting — holding for {self.STARTUP_HOLD}s standstill...')

        self.create_subscription(
            Float32MultiArray, '/plan_cmd', self._cb_plan, 10)

        # Subscribe to raw scan for emergency stop — independent of PLAN node
        self.create_subscription(
            LaserScan, '/scan', self._cb_scan, 10)

        self._pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Publish at 10 Hz (decouple from plan rate for smooth motion)
        self.create_timer(0.1, self._act)

    # ------------------------------------------------------------------
    def _cb_plan(self, msg: Float32MultiArray):
        self._plan_received = True
        heading    = msg.data[0]
        confidence = msg.data[1]

        # Sentinel: goal reached
        if heading == -999.0:
            self._goal_reached = True
            return

        self._goal_reached = False
        self._heading      = float(heading)
        self._confidence   = float(confidence)

    # ------------------------------------------------------------------
    def _cb_scan(self, msg: LaserScan):
        """Emergency stop: halt immediately if any ray is dangerously close."""
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = ranges[(ranges >= msg.range_min) & (ranges <= msg.range_max)]

        if len(valid) == 0:
            return

        closest = float(np.min(valid))
        if closest < self.ESTOP_DISTANCE:
            if not self._estopped:
                self.get_logger().error(
                    f'EMERGENCY STOP — obstacle at {closest:.3f}m '
                    f'(threshold {self.ESTOP_DISTANCE}m). '
                    f'Robot halted. Will resume when path is clear.')
            self._estopped = True
        else:
            if self._estopped:
                self.get_logger().warn(
                    f'Path clear (closest obstacle now {closest:.3f}m). '
                    f'Resuming motion.')
            self._estopped = False

    # ------------------------------------------------------------------
    def _act(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        # ---- startup hold ----
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if elapsed < self.STARTUP_HOLD:
            self._pub.publish(cmd)  # zero velocity
            return
        if elapsed < self.STARTUP_HOLD + 0.15:  # log once at end of hold
            self.get_logger().info('Standstill complete — ACT node ready.')

        # ---- emergency stop (checked independently of plan) ----
        if self._estopped:
            self._pub.publish(cmd)  # zero velocity
            return

        # ---- wait for first plan ----
        if not self._plan_received:
            self._pub.publish(cmd)
            return

        if self._goal_reached:
            self._pub.publish(cmd)
            self.get_logger().info('Holding stop — goal reached', once=True)
            return

        if self._confidence < 0.01:
            cmd.twist.linear.x  = 0.0
            cmd.twist.angular.z = self.ROTATION_SPEED
            self._pub.publish(cmd)
            self.get_logger().debug('Blocked — rotating to find gap')
            return

        # ---- normal forward-bias motion ----
        turn_sharpness = abs(self._heading) / np.pi

        linear = self.FORWARD_SPEED * (1.0 - 0.7 * turn_sharpness)
        linear = float(np.clip(linear, self.MIN_FORWARD, self.MAX_LINEAR))

        angular = self.KP_ANGULAR * self._heading
        angular = float(np.clip(angular, -self.MAX_ANGULAR, self.MAX_ANGULAR))

        cmd.twist.linear.x  = linear
        cmd.twist.angular.z = angular
        self._pub.publish(cmd)

        self.get_logger().debug(
            f'heading={self._heading:.2f}r  '
            f'conf={self._confidence:.2f}  '
            f'lin={linear:.2f}  ang={angular:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = ActNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send a clean stop on shutdown
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        node._pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()