#!/usr/bin/env python3
"""
kpi_recorder_node.py  –  Performance KPI recorder for the anti-poaching system.

Subscribes
----------
  /odom              nav_msgs/Odometry    – drone position & velocity
  /poacher_odom      nav_msgs/Odometry    – poacher position & velocity
                                             (raw local frame – this node
                                             applies the spawn offset itself,
                                             see "Frame note" below)
  /poacher_visible   std_msgs/Bool        – detection state
  /mission_state     std_msgs/String      – TRACK / SEARCH / WAYPOINT
  /brake_event       std_msgs/Bool        – rising-edge from act_node
  /poacher_caught     std_msgs/Bool        – from poacher_detection_node;
                                              on rising edge this node saves
                                              its session immediately and
                                              shuts itself down, so KPI
                                              recording stops at the moment
                                              of capture

Frame note
----------
  /odom and /poacher_odom each reset to (0,0) at their OWN robot's spawn
  point – they are per-robot local odom frames, not a shared world frame.
  This project's convention (matching mission_node.py and
  poacher_detection_node.py) treats the drone's /odom as the reference
  frame, so this node adds (poacher_spawn - drone_spawn) to the poacher's
  raw position before computing distance/speed against the drone. Without
  this offset the first sample reads dist≈0 (both robots' raw (0,0)
  origins coincide) and distance only reflects the poacher's displacement
  from its own spawn point rather than its actual distance from the drone.

Recorded KPIs
-------------
  • Distance between drone and poacher (m)      – 5 Hz time-series
  • Drone speed (m/s)                           – 5 Hz time-series
  • Poacher speed (m/s)                         – 5 Hz time-series
  • Mission state                               – 5 Hz time-series
  • Time-to-first-detection (s)                 – scalar, saved when first seen
    (visibility is ignored for the first `detection_grace_period` seconds
    after node start, so an instantly-visible poacher at t=0 does not
    register as an immediate "detection")
  • Number of emergency brake events             – cumulative counter
  • Number of times poacher was lost             – cumulative counter
    (= transitions from visible → not-visible after at least one detection)
  • Percentage of mission time spent in each phase, plus the two
    higher-level groupings used for reporting:
      - "tracking"    = TRACK           (poacher currently visible, following)
      - "goto_lkp"    = GOTO_LKP        (heading to last-known-position)
      - "pdf_search"  = PDF_SEARCH      (Gaussian diffusion search)
      - "heuristic"   = WAYPOINT_SEARCH (obstacle-fringe waypoint search)
  • poacher_max_speed (m/s)                     – recorded as metadata only
    (not measured – this is the configured Nav2/teleop speed cap for the
    run, passed in as a parameter so it shows up alongside the results
    when comparing iterations with different caps)

Output
------
  Writes a JSON session file on shutdown to:
      ~/Desktop/ROS2/TB3_WS/KPI_Results/session_<YYYY-MM-DD_HH-MM-SS>.json

  The JSON contains:
    - "meta"        : session metadata (start time, software info,
                       configured poacher_max_speed, was_captured flag)
    - "scalars"     : single-value KPIs, incl. time_per_mission_state_s and
                       phase_percentages (tracking / goto_lkp / pdf_search / heuristic)
    - "time_series" : list of per-sample dicts   (t, distance, drone_speed,
                       poacher_speed, mission_state) – the raw data future
                       KPI/plotting scripts (e.g. multi-session averages)
                       should read from here rather than re-deriving anything
    - "events"      : list of timestamped events  (brake, lost, detected,
                       poacher_caught)

Usage
-----
  ros2 run <package> kpi_recorder_node

Parameters
----------
  sample_rate              float  Hz  (default 5.0)
  output_dir                str        (default ~/Desktop/ROS2/TB3_WS/KPI_Results)
  detection_grace_period    float  s   ignore /poacher_visible for this many
                                        seconds after node start (default 2.0)
  poacher_max_speed         float  m/s configured poacher speed cap, recorded
                                        as metadata for cross-run comparison
                                        (default 0.18)
  poacher_spawn_x/y         float  m   poacher's spawn position in the
                                        drone's /odom frame (default 2.0/0.0
                                        – match mission_node.py)
  drone_spawn_x/y           float  m   drone's spawn position (default
                                        -2.0/-0.5 – match mission_node.py)
"""

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


# ── mission_state → reporting phase grouping ──────────────────────────────────
# mission_node.py state strings: 'INIT', 'TRACK', 'GOTO_LKP',
#                                 'PDF_SEARCH', 'WAYPOINT_SEARCH'
PHASE_GROUPS = {
    "tracking":   {"TRACK"},                  # poacher currently visible, following live
    "goto_lkp":   {"GOTO_LKP"},                # heading to last-known-position
    "pdf_search": {"PDF_SEARCH"},             # Gaussian diffusion search
    "heuristic":  {"WAYPOINT_SEARCH"},        # obstacle-fringe waypoint search
}


class KpiRecorderNode(Node):
    """Records performance KPIs and saves a JSON report on shutdown."""

    def __init__(self) -> None:
        super().__init__("kpi_recorder_node")

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter("sample_rate", 5.0)
        # KPI_Results sits at the same level as CE (i.e. inside TB3_WS),
        # not inside the home directory.
        self.declare_parameter(
            "output_dir",
            str(Path.home() / "Desktop" / "ROS2" / "TB3_WS" / "KPI_Results"),
        )
        self.declare_parameter("detection_grace_period", 2.0)
        self.declare_parameter("poacher_max_speed", 0.18)

        # ── world-frame offset for /poacher_odom ────────────────────────────
        # /poacher_odom (and /odom) each reset to (0,0) at their OWN spawn
        # point – per-robot local odom, not a shared world frame. This
        # project's convention (matching mission_node.py and
        # poacher_detection_node.py) treats the drone's /odom as the
        # reference frame, so the poacher's local odom needs this offset
        # added before computing distance against it. Without this, the
        # very first sample reads dist≈0 (both robots' raw (0,0) origins
        # coincide) and distance only reflects the poacher's displacement
        # from ITS OWN spawn point rather than its distance from the drone.
        self.declare_parameter("poacher_spawn_x",  2.0)
        self.declare_parameter("poacher_spawn_y",  0.0)
        self.declare_parameter("drone_spawn_x",   -2.0)
        self.declare_parameter("drone_spawn_y",   -0.5)

        self._rate       = self.get_parameter("sample_rate").value
        self._output_dir = Path(self.get_parameter("output_dir").value)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._grace_period   = self.get_parameter("detection_grace_period").value
        self._poacher_max_speed = self.get_parameter("poacher_max_speed").value
        self._poacher_offset_x = (self.get_parameter("poacher_spawn_x").value
                                   - self.get_parameter("drone_spawn_x").value)
        self._poacher_offset_y = (self.get_parameter("poacher_spawn_y").value
                                   - self.get_parameter("drone_spawn_y").value)

        # ── QoS profiles ─────────────────────────────────────────────────────
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── state ─────────────────────────────────────────────────────────────
        self._start_time: float = time.time()
        self._ros_start:  float | None = None          # set on first odom msg

        # Latest values from topics
        self._drone_x:    float = 0.0
        self._drone_y:    float = 0.0
        self._drone_vx:   float = 0.0
        self._drone_vy:   float = 0.0
        self._drone_ok:   bool  = False

        self._poacher_x:  float = 0.0
        self._poacher_y:  float = 0.0
        self._poacher_vx: float = 0.0
        self._poacher_vy: float = 0.0
        self._poacher_ok: bool  = False

        self._visible:         bool  = False
        self._prev_visible:    bool  = False
        self._ever_detected:   bool  = False
        self._mission_state:   str   = "UNKNOWN"

        # Scalar KPIs
        self._time_to_first_detection: float | None = None
        self._brake_count:  int = 0
        self._lost_count:   int = 0
        self._caught:       bool = False
        self._saved:        bool = False
        self._save_stamp:   str | None = None

        # Time-series & events
        self._samples: list[dict] = []
        self._events:  list[dict] = []

        # ── subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Odometry, "/odom",         self._drone_odom_cb,   best_effort)
        self.create_subscription(
            Odometry, "/poacher_odom", self._poacher_odom_cb, best_effort)
        self.create_subscription(
            Bool,   "/poacher_visible", self._visible_cb,     reliable)
        self.create_subscription(
            String, "/mission_state",   self._state_cb,       reliable)
        self.create_subscription(
            Bool,   "/brake_event",     self._brake_cb,       reliable)
        self.create_subscription(
            Bool,   "/poacher_caught",  self._caught_cb,      reliable)

        # ── sample timer ─────────────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / self._rate, self._sample_cb)

        self.get_logger().info(
            f"KpiRecorderNode started  |  "
            f"rate={self._rate} Hz  |  output={self._output_dir}  |  "
            f"detection_grace_period={self._grace_period} s  |  "
            f"poacher_max_speed={self._poacher_max_speed} m/s"
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _elapsed(self) -> float:
        """Seconds since node started (wall-clock)."""
        return time.time() - self._start_time

    @staticmethod
    def _speed(vx: float, vy: float) -> float:
        return math.hypot(vx, vy)

    # ── topic callbacks ───────────────────────────────────────────────────────
    def _drone_odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self._drone_x, self._drone_y = p.x, p.y
        self._drone_vx, self._drone_vy = v.x, v.y
        self._drone_ok = True
        if self._ros_start is None:
            self._ros_start = self._elapsed()

    def _poacher_odom_cb(self, msg: Odometry) -> None:
        # Raw pose is in the poacher's own LOCAL odom frame (origin at its
        # own spawn point) – convert to the drone's /odom frame, which this
        # project treats as the world/reference frame.
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self._poacher_x = p.x + self._poacher_offset_x
        self._poacher_y = p.y + self._poacher_offset_y
        self._poacher_vx, self._poacher_vy = v.x, v.y
        self._poacher_ok = True

    def _visible_cb(self, msg: Bool) -> None:
        now_visible = msg.data
        t = self._elapsed()

        # Ignore visibility entirely during the startup grace period – an
        # instantly-visible poacher at t≈0 should not register as a
        # legitimate "detection".
        if t < self._grace_period:
            self._prev_visible = now_visible
            return

        # First detection ever → record time-to-detect
        if now_visible and not self._ever_detected:
            self._ever_detected = True
            self._time_to_first_detection = t
            self.get_logger().info(f"[KPI] First detection at t={t:.2f}s")
            self._events.append({"t": t, "event": "first_detection"})

        # Lost after having been visible
        if self._prev_visible and not now_visible and self._ever_detected:
            self._lost_count += 1
            self.get_logger().info(
                f"[KPI] Poacher lost (#{self._lost_count}) at t={t:.2f}s"
            )
            self._events.append({
                "t": t, "event": "poacher_lost", "count": self._lost_count
            })

        # Re-acquired after being lost
        if not self._prev_visible and now_visible and self._ever_detected:
            self._events.append({"t": t, "event": "poacher_reacquired"})

        self._prev_visible = now_visible
        self._visible = now_visible

    def _state_cb(self, msg: String) -> None:
        self._mission_state = msg.data

    def _brake_cb(self, msg: Bool) -> None:
        """act_node publishes True on rising-edge of brake engagement."""
        if msg.data:
            self._brake_count += 1
            t = self._elapsed()
            self.get_logger().info(
                f"[KPI] Emergency brake #{self._brake_count} at t={t:.2f}s"
            )
            self._events.append({
                "t": t, "event": "brake", "count": self._brake_count
            })

    def _caught_cb(self, msg: Bool) -> None:
        """poacher_detection_node publishes True once distance < capture_dist.

        Stops KPI recording immediately: saves the session and shuts this
        node down, rather than continuing to log a frozen, post-capture
        drone for the rest of the run.
        """
        if not msg.data or self._caught:
            return
        self._caught = True
        t = self._elapsed()
        self.get_logger().warn(f"[KPI] POACHER CAUGHT at t={t:.2f}s – stopping recording")
        self._events.append({"t": t, "event": "poacher_caught"})
        self.save()
        # Stop the timer so no further samples are taken, then let main()'s
        # spin() return by shutting down rclpy from here.
        self._timer.cancel()
        rclpy.shutdown()

    # ── periodic sample ───────────────────────────────────────────────────────
    def _sample_cb(self) -> None:
        if not (self._drone_ok and self._poacher_ok):
            return                        # wait until both odoms have arrived

        t = self._elapsed()
        dist = math.hypot(
            self._drone_x - self._poacher_x,
            self._drone_y - self._poacher_y,
        )

        self._samples.append({
            "t":             round(t,    3),
            "distance":      round(dist, 4),
            "drone_speed":   round(self._speed(self._drone_vx,   self._drone_vy),   4),
            "poacher_speed": round(self._speed(self._poacher_vx, self._poacher_vy), 4),
            "poacher_visible": self._visible,
            "mission_state": self._mission_state,
        })

    # ── save ──────────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist full session data to JSON. Called on shutdown (or earlier,
        immediately, if the poacher is caught). Idempotent – calling it twice
        (e.g. once from _caught_cb and once from main()'s finally block) just
        overwrites the same file harmlessly."""
        if self._saved:
            return
        self._saved = True
        stamp = self._save_stamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._save_stamp = stamp
        filename = self._output_dir / f"session_{stamp}.json"

        # Derive summary stats from time-series
        distances = [s["distance"]      for s in self._samples]
        d_speeds  = [s["drone_speed"]   for s in self._samples]
        p_speeds  = [s["poacher_speed"] for s in self._samples]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {}
            return {
                "min":  round(min(vals),  4),
                "max":  round(max(vals),  4),
                "mean": round(sum(vals) / len(vals), 4),
            }

        # Total time in each mission state
        state_time: dict[str, float] = {}
        dt = 1.0 / self._rate
        for s in self._samples:
            st = s["mission_state"]
            state_time[st] = round(state_time.get(st, 0.0) + dt, 3)

        # Percentage of mission time per reporting phase (following / pdf / heuristic)
        total_time = sum(state_time.values())
        phase_time: dict[str, float] = {phase: 0.0 for phase in PHASE_GROUPS}
        for state, dur in state_time.items():
            for phase, members in PHASE_GROUPS.items():
                if state in members:
                    phase_time[phase] += dur

        phase_percentages = {
            phase: round(100.0 * dur / total_time, 2) if total_time > 0 else 0.0
            for phase, dur in phase_time.items()
        }

        payload = {
            "meta": {
                "session_start":      stamp,
                "duration_s":         round(self._elapsed(), 2),
                "sample_rate_hz":     self._rate,
                "total_samples":      len(self._samples),
                "poacher_max_speed":  self._poacher_max_speed,
                "was_captured":       self._caught,
            },
            "scalars": {
                "time_to_first_detection_s": self._time_to_first_detection,
                "brake_events_total":        self._brake_count,
                "times_poacher_lost":        self._lost_count,
                "distance_stats_m":          _stats(distances),
                "drone_speed_stats_mps":     _stats(d_speeds),
                "poacher_speed_stats_mps":   _stats(p_speeds),
                "time_per_mission_state_s":  state_time,
                "time_per_phase_s":          {k: round(v, 3) for k, v in phase_time.items()},
                "phase_percentages":         phase_percentages,
            },
            "events":      self._events,
            "time_series": self._samples,
        }

        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

        self.get_logger().info(
            f"\n[KPI] Session saved → {filename}\n"
            f"  Duration              : {payload['meta']['duration_s']:.1f} s\n"
            f"  Samples               : {len(self._samples)}\n"
            f"  Time-to-detect        : {self._time_to_first_detection}\n"
            f"  Brake events          : {self._brake_count}\n"
            f"  Times lost            : {self._lost_count}\n"
            f"  Distance stats (m)    : {_stats(distances)}\n"
            f"  Drone speed (m/s)     : {_stats(d_speeds)}\n"
            f"  Poacher speed (m/s)   : {_stats(p_speeds)}\n"
            f"  State time (s)        : {state_time}\n"
            f"  Phase % (track/goto_lkp/pdf/heuristic): {phase_percentages}"
        )


# ---------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = KpiRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()                      # no-op if _caught_cb already saved
        node.destroy_node()
        if rclpy.ok():                   # _caught_cb may have already shut down
            rclpy.shutdown()


if __name__ == "__main__":
    main()