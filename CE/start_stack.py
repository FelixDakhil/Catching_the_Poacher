#!/usr/bin/env python3
"""
start_stack.py  –  Bring up the full anti-poaching drone stack in one terminal.

Expected layout (run this script from the CE folder):

  CE/
  ├── start_stack.py
  ├── mission_node.py
  ├── poacher_detection_node.py
  ├── Local_Planner/
  │   └── quick_start.py   (which itself launches see/plan/act/viz from here)
  └── Global_Planner/
      ├── cost_map.py
      └── global_planner_node.py

Launch order
------------
  1. Local_Planner/quick_start.py     – local-planner bundle
                                         (see_node.py -> plan_node.py -> act_node.py -> viz_server.py)
  2. Global_Planner/cost_map.py       – global cost map / occupancy grid
  3. Global_Planner/global_planner_node.py – D* Lite global planner
  4. mission_node.py                  – subsumption mission logic (TRACK / GOTO_LKP / SEARCH)
  5. poacher_detection_node.py        – poacher detection

Each step is staggered so that the topics/TFs a later node depends on have
already been advertised by the time it starts subscribing. Each subprocess
is run with its own subfolder as its working directory, so any relative
paths inside those scripts keep behaving the way they do when run by hand.

quick_start.py manages its own four child processes (see/plan/act/viz) and
normally owns Ctrl+C handling for them. Here it is started as a background
subprocess like everything else, and THIS script becomes the single place
that owns Ctrl+C / shutdown for the whole stack -- so one Ctrl+C tears down
quick_start.py (and in turn its own children) plus the two nodes started
after it.

Usage
-----
  python3 start_stack.py [--dir /path/to/CE]

  --dir   the CE folder containing Local_Planner/, Global_Planner/,
          mission_node.py, and poacher_detection_node.py
          (default: same folder as this script)

Press Ctrl+C once to shut everything down cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


# (subfolder relative to --dir, script, human-readable label,
#  seconds to wait after launch before starting the next step)
#
# Subfolder is "" for scripts that live directly in the CE folder.
STEPS = [
    ("Local_Planner",  "quick_start.py",             "local planner bundle (see/plan/act/viz)", 2.0),
    ("Global_Planner", "cost_map.py",                "global cost map",                         1.0),
    ("Global_Planner", "global_planner_node.py",     "global planner (D* Lite)",                1.0),
    ("",               "mission_node.py",            "mission node",                            1.0),
    ("",               "poacher_detection_node.py",  "poacher detection",                       0.0),
]

# Colour helpers (no external deps)
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"


def tag(name: str) -> str:
    return f"[{name}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the full anti-poaching stack in order.")
    parser.add_argument(
        "--dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing the node scripts (default: script's own directory)",
    )
    args = parser.parse_args()

    node_dir = os.path.abspath(args.dir)

    # Verify every script exists before starting anything
    for subdir, script, _label, _delay in STEPS:
        path = os.path.join(node_dir, subdir, script)
        if not os.path.isfile(path):
            print(f"{_RED}ERROR: cannot find {path}{_RESET}", file=sys.stderr)
            sys.exit(1)

    processes: list[subprocess.Popen] = []
    names: list[str] = []

    def shutdown(signum=None, frame=None) -> None:
        print(f"\n{_YELLOW}Shutting down all nodes…{_RESET}")
        # Shut down in reverse order so downstream consumers stop before
        # the nodes they depend on.
        for proc in reversed(processes):
            if proc.poll() is None:          # still running
                proc.send_signal(signal.SIGINT)
        # Give them a moment to exit gracefully (quick_start.py itself
        # needs time to tear down its own four children)
        time.sleep(2.0)
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.kill()
        print(f"{_GREEN}All nodes stopped.{_RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"{_GREEN}Launching stack from: {node_dir}{_RESET}\n")

    for subdir, script, label, delay in STEPS:
        script_dir = os.path.join(node_dir, subdir) if subdir else node_dir
        path = os.path.join(script_dir, script)
        display = f"{subdir}/{script}" if subdir else script
        print(f"  {_GREEN}▶{_RESET}  {tag(display)}  ({label})")
        proc = subprocess.Popen(
            [sys.executable, path],
            cwd=script_dir,
            # Each process keeps its own stdout/stderr so logs stay visible
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(proc)
        names.append(display)
        if delay > 0:
            time.sleep(delay)

    print(f"\n{_GREEN}All nodes running.{_RESET}  Press Ctrl+C to stop.\n")

    # Monitor: if any step dies unexpectedly, report it and tear everything down
    while True:
        time.sleep(1.0)
        for proc, name in zip(processes, names):
            rc = proc.poll()
            if rc is not None:
                print(
                    f"{_RED}{tag(name)} exited unexpectedly "
                    f"(return code {rc}){_RESET}",
                    file=sys.stderr,
                )
                shutdown()


if __name__ == "__main__":
    main()