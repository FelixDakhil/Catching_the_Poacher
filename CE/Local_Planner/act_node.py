#!/usr/bin/env python3
"""
act_node.py  –  Actuation layer for TurtleBot3 in Gazebo.

Responsibilities
----------------
* Subscribe to /vfh_command  (Float64MultiArray  [linear_vel, angular_vel])
  produced by plan_node.
* Subscribe to /processed_scan to enforce a front-clearance emergency brake.
  If any beam within BRAKE_CONE_DEG degrees of dead-ahead is closer than
  BRAKE_DIST_M, linear velocity is forced to zero. Angular velocity is
  ALWAYS passed through unchanged so the planner can steer clear.
* Apply hardware clamps (TurtleBot3 Burger limits).
* Watchdog: publish a zero-velocity stop if the planner goes silent for
  more than `watchdog_timeout` seconds. Watchdog is suspended while the
  brake is active so angular velocity is preserved during obstacle avoidance.
* Publish geometry_msgs/TwistStamped on /cmd_vel at a fixed rate (default 10 Hz).

Note
----
  Capture (/poacher_caught from poacher_detection_node) is NOT handled here.
  It's purely a reporting signal – kpi_recorder_node listens for it to stop
  recording, but it has no effect on actuation. The drone keeps moving
  under the planner's commands even after capture; nothing here brakes it.

Parameters
----------
  publish_rate      float  Hz for the output timer          (default 10.0)
  watchdog_timeout  float  seconds before emergency stop    (default 0.5)
  use_stamped       bool   True → TwistStamped, False → Twist (default True)
  base_frame        str    header frame_id                  (default "base_footprint")
  max_linear_vel    float  m/s clamp                        (default 0.22)
  max_angular_vel   float  rad/s clamp                       (default 2.84)
  brake_dist        float  m  front-clearance threshold      (default 0.25)
  brake_cone_deg    float  half-angle of front cone (deg)    (default 30.0)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped, Twist
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray


# ---------------------------------------------------------------------------
# Hardware limits – TurtleBot3 Burger
# ---------------------------------------------------------------------------
MAX_LINEAR_VEL  = 0.22   # m/s
MAX_ANGULAR_VEL = 2.84   # rad/s

# Emergency-brake defaults
BRAKE_DIST_M    = 0.15   # metres – minimum front clearance
BRAKE_CONE_DEG  = 30.0   # half-angle of the danger cone ahead of the robot

DEFAULT_RATE     = 10.0  # Hz
DEFAULT_WATCHDOG = 0.5   # seconds


class ActNode(Node):
    """Actuation node: /vfh_command → /cmd_vel TwistStamped with safety brake."""

    def __init__(self) -> None:
        super().__init__("act_node")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("publish_rate",     DEFAULT_RATE)
        self.declare_parameter("watchdog_timeout", DEFAULT_WATCHDOG)
        self.declare_parameter("use_stamped",      True)
        self.declare_parameter("base_frame",       "base_footprint")
        self.declare_parameter("max_linear_vel",   MAX_LINEAR_VEL)
        self.declare_parameter("max_angular_vel",  MAX_ANGULAR_VEL)
        self.declare_parameter("brake_dist",       BRAKE_DIST_M)
        self.declare_parameter("brake_cone_deg",   BRAKE_CONE_DEG)

        self._rate       = self.get_parameter("publish_rate").value
        self._watchdog   = self.get_parameter("watchdog_timeout").value
        self._stamped    = self.get_parameter("use_stamped").value
        self._frame      = self.get_parameter("base_frame").value
        self._max_lin    = self.get_parameter("max_linear_vel").value
        self._max_ang    = self.get_parameter("max_angular_vel").value
        self._brake_d    = self.get_parameter("brake_dist").value
        self._brake_cone = math.radians(self.get_parameter("brake_cone_deg").value)

        # ---- internal state -----------------------------------------------
        self._target_linear:  float       = 0.0
        self._target_angular: float       = 0.0
        self._last_cmd_time:  Time | None = None
        self._brake_active:   bool        = False

        # Scan geometry
        self._angle_min:     float = 0.0
        self._angle_inc:     float = math.radians(1.0)
        self._meta_received: bool  = False

        # ---- QoS ----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers --------------------------------------------------
        self.create_subscription(
            Float64MultiArray, "/vfh_command",    self._command_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/processed_scan", self._scan_cb, sensor_qos)
        self.create_subscription(
            Float64MultiArray, "/scan_metadata",  self._meta_cb, 10)

        # ---- publishers ---------------------------------------------------
        if self._stamped:
            self._vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Rising-edge brake event – True when brake first engages.
        # kpi_recorder_node listens to this to count brake events.
        self._brake_event_pub = self.create_publisher(Bool, "/brake_event", 10)

        self._timer = self.create_timer(1.0 / self._rate, self._publish_cb)

        self.get_logger().info(
            f"ActNode ready  |  "
            f"{'TwistStamped' if self._stamped else 'Twist'} on /cmd_vel  "
            f"@ {self._rate} Hz  |  watchdog={self._watchdog} s  |  "
            f"brake_dist={self._brake_d} m  brake_cone=±{math.degrees(self._brake_cone):.0f}°"
        )

    # -----------------------------------------------------------------------
    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._angle_min     = msg.data[0]
            self._angle_inc     = msg.data[2]
            self._meta_received = True

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        """Check front clearance. Only sets _brake_active – never touches velocities."""
        if not self._meta_received:
            return

        ranges = msg.data
        if not ranges:
            return

        blocked = any(
            r < self._brake_d
            for i, r in enumerate(ranges)
            if abs(math.atan2(
                math.sin(self._angle_min + i * self._angle_inc),
                math.cos(self._angle_min + i * self._angle_inc)
            )) <= self._brake_cone
        )

        if blocked and not self._brake_active:
            self.get_logger().warn(
                f"BRAKE: obstacle < {self._brake_d} m "
                f"in ±{math.degrees(self._brake_cone):.0f}° cone – lin zeroed"
            )
            # Rising edge → notify KPI recorder
            self._brake_event_pub.publish(Bool(data=True))
        elif not blocked and self._brake_active:
            self.get_logger().info("BRAKE released – forward motion restored")

        self._brake_active = blocked

    def _command_cb(self, msg: Float64MultiArray) -> None:
        """Store planner command. Hardware clamp applied here."""
        if len(msg.data) < 2:
            return
        self._target_linear  = float(max(-self._max_lin, min(self._max_lin,  msg.data[0])))
        self._target_angular = float(max(-self._max_ang, min(self._max_ang, msg.data[1])))
        self._last_cmd_time  = self.get_clock().now()

    # -----------------------------------------------------------------------
    def _publish_cb(self) -> None:
        now = self.get_clock().now()

        # Watchdog: only fires when brake is NOT active.
        # While braking the robot must keep turning, so we preserve angular.
        if not self._brake_active and self._last_cmd_time is not None:
            elapsed = (now - self._last_cmd_time).nanoseconds * 1e-9
            if elapsed > self._watchdog:
                if self._target_linear != 0.0 or self._target_angular != 0.0:
                    self.get_logger().warn(
                        f"Watchdog ({elapsed:.2f} s without command) – stop"
                    )
                self._target_linear  = 0.0
                self._target_angular = 0.0

        # Brake: zero linear only. Angular always passes through.
        lin = 0.0 if self._brake_active else self._target_linear
        ang = self._target_angular

        self._publish(now, lin, ang)

    def _publish(self, now: Time, lin: float, ang: float) -> None:
        if self._stamped:
            msg = TwistStamped()
            msg.header.stamp    = now.to_msg()
            msg.header.frame_id = self._frame
            msg.twist.linear.x  = lin
            msg.twist.linear.y  = 0.0
            msg.twist.linear.z  = 0.0
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = ang
        else:
            msg = Twist()
            msg.linear.x  = lin
            msg.angular.z = ang
        self._vel_pub.publish(msg)

    def stop(self) -> None:
        self._target_linear  = 0.0
        self._target_angular = 0.0
        self._publish(self.get_clock().now(), 0.0, 0.0)
        self.get_logger().info("ActNode: stop issued")


# ---------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()