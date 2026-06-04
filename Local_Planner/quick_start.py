#!/usr/bin/env python3
"""
launch_all.py  –  Start all TurtleBot3 nodes in one terminal.

Runs:
  1. see_node.py      – perception / LiDAR cleaning
  2. plan_node.py     – VFH+ local planner
  3. act_node.py      – actuation + safety brake
  4. viz_server.py    – WebSocket server for the LiDAR dashboard

Usage
-----
  python3 launch_all.py [--dir /path/to/nodes]

  --dir   folder containing the four .py files (default: same folder as
          this script)

Press Ctrl+C once to shut everything down cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


NODES = [
    "see_node.py",
    "plan_node.py",
    "act_node.py",
    "viz_server.py",
]

# Colour helpers (no external deps)
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"

def tag(name: str) -> str:
    return f"[{name}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch all TurtleBot3 nodes.")
    parser.add_argument(
        "--dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing the node scripts (default: script's own directory)",
    )
    args = parser.parse_args()

    node_dir = os.path.abspath(args.dir)

    # Verify every file exists before starting anything
    for node in NODES:
        path = os.path.join(node_dir, node)
        if not os.path.isfile(path):
            print(f"{_RED}ERROR: cannot find {path}{_RESET}", file=sys.stderr)
            sys.exit(1)

    processes: list[subprocess.Popen] = []

    def shutdown(signum=None, frame=None) -> None:
        print(f"\n{_YELLOW}Shutting down all nodes…{_RESET}")
        for proc in processes:
            if proc.poll() is None:          # still running
                proc.send_signal(signal.SIGINT)
        # Give them a moment to exit gracefully
        time.sleep(1.5)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
        print(f"{_GREEN}All nodes stopped.{_RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"{_GREEN}Launching nodes from: {node_dir}{_RESET}\n")

    for node in NODES:
        path = os.path.join(node_dir, node)
        print(f"  {_GREEN}▶{_RESET}  {tag(node)}")
        proc = subprocess.Popen(
            [sys.executable, path],
            # Each node keeps its own stdout/stderr so logs stay visible
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(proc)
        # Small stagger: give see_node a moment to advertise /scan_metadata
        # before plan_node and act_node start subscribing
        time.sleep(0.4)

    print(f"\n{_GREEN}All nodes running.{_RESET}  Press Ctrl+C to stop.\n")

    # Monitor: if any node dies unexpectedly, report it
    while True:
        time.sleep(1.0)
        for proc, node in zip(processes, NODES):
            rc = proc.poll()
            if rc is not None:
                print(
                    f"{_RED}{tag(node)} exited unexpectedly "
                    f"(return code {rc}){_RESET}",
                    file=sys.stderr,
                )
                shutdown()


if __name__ == "__main__":
    main()