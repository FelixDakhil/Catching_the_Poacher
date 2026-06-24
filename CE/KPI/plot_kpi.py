#!/usr/bin/env python3
"""
plot_kpi.py  –  Load a KPI session JSON and produce a multi-panel report.

Usage
-----
  python3 plot_kpi.py                         # opens file picker / uses latest
  python3 plot_kpi.py ~/Desktop/ROS2/TB3_WS/KPI_Results/session_X.json

Output
------
  A PNG alongside the JSON file, e.g. session_X.png
  Also prints a plain-text summary to stdout.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_DRONE   = "#2196F3"
C_POACHER = "#E91E63"
C_DIST    = "#4CAF50"
C_BRAKE   = "#FF5722"
C_LOST    = "#9C27B0"
C_DETECT  = "#00BCD4"

STATE_COLORS = {
    "TRACK":           "#2196F3",
    "GOTO_LKP":        "#3F51B5",
    "PDF_SEARCH":      "#FF9800",
    "WAYPOINT_SEARCH": "#9E9E9E",
    "INIT":            "#EEEEEE",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _latest(folder: Path) -> Path:
    files = sorted(folder.glob("session_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No session files found in {folder}")
    return files[-1]


def _shade_states(ax, times, states):
    """Draw coloured background bands for each mission state run."""
    if not times:
        return
    prev_state = states[0]
    prev_t     = times[0]
    for t, s in zip(times[1:], states[1:]):
        if s != prev_state:
            ax.axvspan(prev_t, t, alpha=0.12,
                       color=STATE_COLORS.get(prev_state, "#EEEEEE"), lw=0)
            prev_state = s
            prev_t = t
    ax.axvspan(prev_t, times[-1], alpha=0.12,
               color=STATE_COLORS.get(prev_state, "#EEEEEE"), lw=0)


def _event_vlines(ax, events, kind, color, label, ymin=0.0, ymax=1.0):
    first = True
    for ev in events:
        if ev["event"] == kind:
            ax.axvline(ev["t"], color=color, lw=1.2, ls="--",
                       label=label if first else None,
                       ymin=ymin, ymax=ymax)
            first = False


def plot_distance_only(path: Path, data: dict | None = None) -> Path:
    """
    Standalone single-axes distance-vs-time plot for one session.

    Kept separate from the multi-panel report so a future multi-session
    script can import this (or copy its plotting calls) to overlay several
    sessions on one axes, or average several `time_series` arrays together
    before plotting — the raw per-sample distance values used here come
    straight from session["time_series"], so nothing needs to be
    re-derived from the recorder.

    Returns the path to the saved PNG.
    """
    if data is None:
        data = _load(path)

    ts = data["time_series"]
    if not ts:
        raise ValueError("No time-series samples in file – nothing to plot.")

    t    = np.array([s["t"]        for s in ts])
    dist = np.array([s["distance"] for s in ts])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, dist, color=C_DIST, lw=1.6, label="Distance (m)")

    for ev in data["events"]:
        if ev["event"] == "first_detection":
            ax.axvline(ev["t"], color=C_DETECT, lw=1.2, ls="--", label="First detection")
        elif ev["event"] == "poacher_lost":
            ax.axvline(ev["t"], color=C_LOST, lw=1.0, ls=":", alpha=0.8,
                       label="Poacher lost" if "Poacher lost" not in
                       [h.get_label() for h in ax.get_lines()] else None)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Drone ↔ Poacher distance — {data['meta']['session_start']}")
    ax.grid(True, lw=0.4, alpha=0.5)

    # de-duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)

    out = path.with_name(path.stem + "_distance.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def plot(path: Path) -> None:
    data    = _load(path)
    meta    = data["meta"]
    scalars = data["scalars"]
    events  = data["events"]
    ts      = data["time_series"]

    if not ts:
        print("No time-series samples in file – nothing to plot.")
        return

    t      = np.array([s["t"]             for s in ts])
    dist   = np.array([s["distance"]      for s in ts])
    d_spd  = np.array([s["drone_speed"]   for s in ts])
    p_spd  = np.array([s["poacher_speed"] for s in ts])
    states = [s["mission_state"]          for s in ts]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Anti-Poaching System – KPI Report\n"
        f"Session: {meta['session_start']}   "
        f"Duration: {meta['duration_s']:.1f} s   "
        f"Samples: {meta['total_samples']}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ── panel 1: distance ─────────────────────────────────────────────────────
    ax1 = axes[0]
    _shade_states(ax1, t.tolist(), states)
    ax1.plot(t, dist, color=C_DIST, lw=1.4, label="Distance (m)")
    _event_vlines(ax1, events, "first_detection", C_DETECT, "First detection")
    _event_vlines(ax1, events, "poacher_lost",    C_LOST,   "Poacher lost")
    _event_vlines(ax1, events, "brake",           C_BRAKE,  "Brake event")
    ax1.set_ylabel("Distance (m)", fontsize=9)
    ax1.set_title("Drone ↔ Poacher distance", fontsize=9, loc="left")
    ax1.legend(loc="upper right", fontsize=7, ncol=4)
    ax1.grid(True, lw=0.4, alpha=0.5)

    # ── panel 2: speeds ───────────────────────────────────────────────────────
    ax2 = axes[1]
    _shade_states(ax2, t.tolist(), states)
    ax2.plot(t, d_spd, color=C_DRONE,   lw=1.3, label="Drone speed (m/s)")
    ax2.plot(t, p_spd, color=C_POACHER, lw=1.3, ls="--", label="Poacher speed (m/s)")
    _event_vlines(ax2, events, "brake",        C_BRAKE, "Brake event")
    _event_vlines(ax2, events, "poacher_lost", C_LOST,  "Poacher lost")
    ax2.set_ylabel("Speed (m/s)", fontsize=9)
    ax2.set_title("Velocities", fontsize=9, loc="left")
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, lw=0.4, alpha=0.5)

    # ── panel 3: mission state ────────────────────────────────────────────────
    ax3 = axes[2]
    state_vals = {"TRACK": 3, "GOTO_LKP": 2, "PDF_SEARCH": 1, "WAYPOINT_SEARCH": 0, "INIT": -1}
    y_state = [state_vals.get(s, -1) for s in states]
    _shade_states(ax3, t.tolist(), states)
    ax3.step(t, y_state, where="post", color="#333", lw=1.0)
    ax3.set_yticks([3, 2, 1, 0])
    ax3.set_yticklabels(["TRACK", "GOTO_LKP", "PDF_SEARCH", "WAYPOINT_SEARCH"], fontsize=8)
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_title("Mission state", fontsize=9, loc="left")
    ax3.grid(True, lw=0.4, alpha=0.5, axis="x")

    # State legend patches
    patches = [
        mpatches.Patch(color=c, alpha=0.4, label=s)
        for s, c in STATE_COLORS.items() if s != "INIT"
    ]
    ax3.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)

    # ── scalar annotations (text box) ─────────────────────────────────────────
    ttd = scalars.get("time_to_first_detection_s")
    ttd_str = f"{ttd:.2f} s" if ttd is not None else "never"
    phase_pct = scalars.get("phase_percentages", {})
    summary_lines = [
        f"Time to 1st detection : {ttd_str}",
        f"Brake events          : {scalars['brake_events_total']}",
        f"Times poacher lost    : {scalars['times_poacher_lost']}",
        f"Dist mean / max (m)   : "
        f"{scalars['distance_stats_m'].get('mean', '–')} / "
        f"{scalars['distance_stats_m'].get('max', '–')}",
        f"Drone mean speed (m/s): {scalars['drone_speed_stats_mps'].get('mean', '–')}",
        f"Poacher mean spd(m/s) : {scalars['poacher_speed_stats_mps'].get('mean', '–')}",
        f"Following %           : {phase_pct.get('following', '–')}",
        f"PDF search %          : {phase_pct.get('pdf_search', '–')}",
        f"Heuristic search %    : {phase_pct.get('heuristic', '–')}",
    ]
    fig.text(
        0.01, 0.01, "\n".join(summary_lines),
        fontsize=7, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out}")

    dist_out = plot_distance_only(path, data=data)
    print(f"Saved standalone distance plot → {dist_out}")

    # ── plain-text summary ────────────────────────────────────────────────────
    print("\n── KPI Summary ─────────────────────────────────────────────")
    for line in summary_lines:
        print(" ", line)
    print(f"  State time breakdown (s):")
    for state, dur in scalars.get("time_per_mission_state_s", {}).items():
        print(f"    {state:<10} {dur:.1f} s")
    print(f"  Phase breakdown (%):")
    for phase, pct in phase_pct.items():
        print(f"    {phase:<12} {pct:.1f} %")
    print("────────────────────────────────────────────────────────────\n")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
    else:
        # KPI_Results sits at the same level as CE (i.e. inside TB3_WS)
        folder = Path.home() / "Desktop" / "ROS2" / "TB3_WS" / "KPI_Results"
        path   = _latest(folder)
        print(f"Auto-selected latest session: {path}")

    plot(path)