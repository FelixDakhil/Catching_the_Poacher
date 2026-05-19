#!/usr/bin/env python3
"""
act_node.py  –  Actuation layer for TurtleBot3 in Gazebo.

Responsibilities
----------------
* Subscribe to /vfh_command  (Float64MultiArray  [linear_vel, angular_vel])
  produced by plan_node.
* Apply hardware safety limits (clamp to TurtleBot3 Burger specs).
* Apply a watchdog: if no command arrives within `watchdog_timeout` seconds,
  publish a zero-velocity stop command.
* Publish geometry_msgs/TwistStamped on /cmd_vel at a configurable rate
  (default 10 Hz) so the Gazebo differential-drive plugin keeps receiving
  commands even between planner cycles.

Why TwistStamped?
-----------------
ROS 2 Humble / Nav2 Jazzy uses TwistStamped on /cmd_vel by default.
The TurtleBot3 Gazebo sim uses the same convention when launched with
the nav2 bringup.  If your sim uses the older plain Twist topic, set
the `use_stamped` parameter to False (see parameters below).
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TwistStamped, Twist
from std_msgs.msg import Float64MultiArray


# ---------------------------------------------------------------------------
# Hardware limits – TurtleBot3 Burger
# ---------------------------------------------------------------------------
MAX_LINEAR_VEL  = 0.22   # m/s
MAX_ANGULAR_VEL = 2.84   # rad/s

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PUBLISH_RATE    = 10.0   # Hz
DEFAULT_WATCHDOG_TIMEOUT = 0.5   # seconds – stop if no command received


class ActNode(Node):
    """Actuation node: converts planner output to /cmd_vel TwistStamped."""

    def __init__(self) -> None:
        super().__init__("act_node")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("publish_rate",     DEFAULT_PUBLISH_RATE)
        self.declare_parameter("watchdog_timeout", DEFAULT_WATCHDOG_TIMEOUT)
        self.declare_parameter("use_stamped",      True)   # False → plain Twist
        self.declare_parameter("base_frame",       "base_footprint")
        self.declare_parameter("max_linear_vel",   MAX_LINEAR_VEL)
        self.declare_parameter("max_angular_vel",  MAX_ANGULAR_VEL)

        self._rate      = self.get_parameter("publish_rate").value
        self._watchdog  = self.get_parameter("watchdog_timeout").value
        self._stamped   = self.get_parameter("use_stamped").value
        self._frame     = self.get_parameter("base_frame").value
        self._max_lin   = self.get_parameter("max_linear_vel").value
        self._max_ang   = self.get_parameter("max_angular_vel").value

        # ---- state --------------------------------------------------------
        self._target_linear:  float = 0.0
        self._target_angular: float = 0.0
        self._last_cmd_time: Time | None = None   # wall-clock of last command

        # ---- subscriber ---------------------------------------------------
        self._cmd_sub = self.create_subscription(
            Float64MultiArray,
            "/vfh_command",
            self._command_cb,
            10,
        )

        # ---- publisher ----------------------------------------------------
        if self._stamped:
            self._vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ---- periodic publish timer ---------------------------------------
        period = 1.0 / self._rate
        self._timer = self.create_timer(period, self._publish_cb)

        self.get_logger().info(
            f"ActNode ready  |  publishing {'TwistStamped' if self._stamped else 'Twist'} "
            f"on /cmd_vel  @  {self._rate} Hz  |  watchdog={self._watchdog} s"
        )

    # -----------------------------------------------------------------------
    # Command subscriber
    # -----------------------------------------------------------------------
    def _command_cb(self, msg: Float64MultiArray) -> None:
        """Receive [linear_vel, angular_vel] from the planner."""
        if len(msg.data) < 2:
            self.get_logger().warn("Received malformed vfh_command (need 2 floats)")
            return

        # Safety clamp
        self._target_linear  = float(max(-self._max_lin,  min(self._max_lin,  msg.data[0])))
        self._target_angular = float(max(-self._max_ang, min(self._max_ang, msg.data[1])))

        self._last_cmd_time = self.get_clock().now()

        self.get_logger().debug(
            f"Received vfh_command: v={msg.data[0]:.3f}→{self._target_linear:.3f}  "
            f"ω={msg.data[1]:.3f}→{self._target_angular:.3f}"
        )

    # -----------------------------------------------------------------------
    # Publish timer
    # -----------------------------------------------------------------------
    def _publish_cb(self) -> None:
        """Called at `publish_rate` Hz; applies watchdog and sends /cmd_vel."""
        now = self.get_clock().now()

        # Watchdog: zero velocity if planner has gone silent
        if self._last_cmd_time is not None:
            elapsed = (now - self._last_cmd_time).nanoseconds * 1e-9
            if elapsed > self._watchdog:
                if self._target_linear != 0.0 or self._target_angular != 0.0:
                    self.get_logger().warn(
                        f"Watchdog triggered ({elapsed:.2f} s without command) – "
                        "sending stop"
                    )
                self._target_linear  = 0.0
                self._target_angular = 0.0

        self._publish(now)

    def _publish(self, now: Time) -> None:
        """Build and publish the velocity message."""
        if self._stamped:
            msg = TwistStamped()
            msg.header.stamp    = now.to_msg()
            msg.header.frame_id = self._frame
            msg.twist.linear.x  = self._target_linear
            msg.twist.linear.y  = 0.0
            msg.twist.linear.z  = 0.0
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = self._target_angular
        else:
            msg = Twist()
            msg.linear.x  = self._target_linear
            msg.angular.z = self._target_angular

        self._vel_pub.publish(msg)

    # -----------------------------------------------------------------------
    # Convenience: emergency stop (can be called externally if needed)
    # -----------------------------------------------------------------------
    def stop(self) -> None:
        """Immediately zero velocities and publish."""
        self._target_linear  = 0.0
        self._target_angular = 0.0
        self._publish(self.get_clock().now())
        self.get_logger().info("ActNode: emergency stop issued")


# ---------------------------------------------------------------------------
# Entry-point
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
