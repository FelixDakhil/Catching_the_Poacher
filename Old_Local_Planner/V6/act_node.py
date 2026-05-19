#!/usr/bin/env python3
"""
act_node.py  –  Actuation layer for TurtleBot3 in Gazebo.

Responsibilities
----------------
* Subscribe to /vfh_command  (Float64MultiArray  [linear_vel, angular_vel])
  produced by plan_node.
* Subscribe to /processed_scan to enforce a front-clearance emergency brake.
  If any beam within BRAKE_CONE_DEG degrees of dead-ahead is closer than
  BRAKE_DIST_M, linear velocity is forced to zero regardless of the planner.
* Apply hardware clamps (TurtleBot3 Burger limits).
* Watchdog: publish a zero-velocity stop if the planner goes silent for
  more than `watchdog_timeout` seconds.
* Publish geometry_msgs/TwistStamped on /cmd_vel at a fixed rate (default 10 Hz).

Parameters
----------
  publish_rate      float  Hz for the output timer          (default 10.0)
  watchdog_timeout  float  seconds before emergency stop    (default 0.5)
  use_stamped       bool   True → TwistStamped, False → Twist (default True)
  base_frame        str    header frame_id                  (default "base_footprint")
  max_linear_vel    float  m/s clamp                        (default 0.22)
  max_angular_vel   float  rad/s clamp                      (default 2.84)
  brake_dist        float  m  front-clearance threshold     (default 0.25)
  brake_cone_deg    float  half-angle of front cone (deg)   (default 30.0)

Fixes applied
-------------
  FIX-3: Emergency brake is no longer gated on _meta_received.  When
         metadata has not yet arrived we fall back to conservative defaults
         (full 360° scan assumed at 1° increments) so the brake is active
         from the very first scan message.
  FIX-4: BRAKE_DIST_M constant corrected from 0.15 m → 0.25 m to match
         the documented default and give the 10 Hz loop enough margin to
         stop before a collision.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped, Twist
from std_msgs.msg import Float32MultiArray, Float64MultiArray


# ---------------------------------------------------------------------------
# Hardware limits – TurtleBot3 Burger
# ---------------------------------------------------------------------------
MAX_LINEAR_VEL  = 0.22   # m/s
MAX_ANGULAR_VEL = 2.84   # rad/s

# Emergency-brake defaults
# FIX-4: was 0.15 m – too small for the 10 Hz publish loop at 0.15 m/s.
# The robot travels ~0.015 m per cycle, so at 0.15 m it has already
# committed to the collision.  0.25 m matches the docstring and gives
# ~1.5 cycles of braking margin.
BRAKE_DIST_M    = 0.1   # metres – minimum front clearance
BRAKE_CONE_DEG  = 45.0   # half-angle of the danger cone ahead of the robot

DEFAULT_RATE     = 10.0   # Hz
DEFAULT_WATCHDOG = 0.5    # seconds

# FIX-3: conservative fallback beam geometry used before /scan_metadata arrives
FALLBACK_ANGLE_MIN = -math.pi          # assume full 360° scan
FALLBACK_ANGLE_INC = math.radians(1.0) # 1° per beam (LDS-01 nominal)


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

        # Front-clearance state (updated by scan callback)
        self._front_clear: bool = True   # assume clear until proven otherwise

        # Scan geometry (updated by /scan_metadata when it arrives).
        # FIX-3: initialised to conservative fallback values so the brake
        # evaluates correctly even before the first metadata message.
        self._angle_min:     float = FALLBACK_ANGLE_MIN
        self._angle_inc:     float = FALLBACK_ANGLE_INC
        self._meta_received: bool  = False   # informational only – no longer a gate

        # ---- QoS to match see_node BEST_EFFORT publisher ------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers --------------------------------------------------
        self._cmd_sub = self.create_subscription(
            Float64MultiArray,
            "/vfh_command",
            self._command_cb,
            10,
        )
        self._scan_sub = self.create_subscription(
            Float32MultiArray,
            "/processed_scan",
            self._scan_cb,
            sensor_qos,
        )
        # Grab beam geometry from see_node
        self._meta_sub = self.create_subscription(
            Float64MultiArray,
            "/scan_metadata",
            self._meta_cb,
            10,
        )

        # ---- publisher ----------------------------------------------------
        if self._stamped:
            self._vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ---- periodic publish timer ---------------------------------------
        self._timer = self.create_timer(1.0 / self._rate, self._publish_cb)

        self.get_logger().info(
            f"ActNode ready  |  "
            f"{'TwistStamped' if self._stamped else 'Twist'} on /cmd_vel  "
            f"@ {self._rate} Hz  |  watchdog={self._watchdog} s  |  "
            f"brake_dist={self._brake_d} m  brake_cone=±{math.degrees(self._brake_cone):.0f}°  |  "
            f"brake active from first scan (fallback geometry until /scan_metadata)"
        )

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------
    def _meta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._angle_min     = msg.data[0]
            self._angle_inc     = msg.data[2]
            if not self._meta_received:
                self.get_logger().info(
                    f"Scan metadata received – brake geometry updated "
                    f"(angle_min={math.degrees(self._angle_min):.1f}°, "
                    f"angle_inc={math.degrees(self._angle_inc):.2f}°)"
                )
            self._meta_received = True

    def _scan_cb(self, msg: Float32MultiArray) -> None:
        """
        Evaluate front-clearance emergency brake.

        Scans the beams within ±brake_cone_deg of dead-ahead (angle ≈ 0 in
        robot frame).  If any beam is closer than brake_dist, block forward
        motion.

        FIX-3: The metadata gate has been removed.  Before /scan_metadata
        arrives we use FALLBACK_ANGLE_MIN / FALLBACK_ANGLE_INC so the brake
        is active from the very first scan message instead of being silently
        skipped.
        """
        ranges = msg.data
        if not ranges:
            return

        clear = True
        for i, r in enumerate(ranges):
            beam_angle = self._angle_min + i * self._angle_inc
            # Normalise to [-pi, pi]
            beam_angle = math.atan2(math.sin(beam_angle), math.cos(beam_angle))

            if abs(beam_angle) <= self._brake_cone:
                if r < self._brake_d:
                    clear = False
                    break

        if self._front_clear != clear:
            if not clear:
                self.get_logger().warn(
                    f"EMERGENCY BRAKE: obstacle within {self._brake_d} m "
                    f"in front ±{math.degrees(self._brake_cone):.0f}°"
                )
            else:
                self.get_logger().info("Front clearance restored – brake released")
        self._front_clear = clear

    def _command_cb(self, msg: Float64MultiArray) -> None:
        """Receive [linear_vel, angular_vel] from the planner."""
        if len(msg.data) < 2:
            self.get_logger().warn("Malformed /vfh_command (need 2 floats)")
            return

        # Hardware clamp
        self._target_linear  = float(
            max(-self._max_lin, min(self._max_lin,  msg.data[0]))
        )
        self._target_angular = float(
            max(-self._max_ang, min(self._max_ang, msg.data[1]))
        )
        self._last_cmd_time  = self.get_clock().now()

        self.get_logger().debug(
            f"cmd  v={self._target_linear:.3f}  ω={self._target_angular:.3f}"
        )

    # -----------------------------------------------------------------------
    # Publish timer
    # -----------------------------------------------------------------------
    def _publish_cb(self) -> None:
        now = self.get_clock().now()

        # Watchdog: zero velocity if planner is silent
        if self._last_cmd_time is not None:
            elapsed = (now - self._last_cmd_time).nanoseconds * 1e-9
            if elapsed > self._watchdog:
                if self._target_linear != 0.0 or self._target_angular != 0.0:
                    self.get_logger().warn(
                        f"Watchdog ({elapsed:.2f} s without command) – stop"
                    )
                self._target_linear  = 0.0
                self._target_angular = 0.0

        # Emergency brake: suppress forward motion if too close ahead
        lin = self._target_linear
        ang = self._target_angular
        if not self._front_clear and lin > 0.0:
            lin = 0.0   # kill forward speed; allow steering to escape

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

    # -----------------------------------------------------------------------
    def stop(self) -> None:
        """Immediately zero velocities and publish (called on KeyboardInterrupt)."""
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