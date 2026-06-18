#!/usr/bin/env python3
"""Standalone PICO XRoboToolkit -> Axol VRFrame bridge.

This wrapper is intentionally tiny so it can be executed from an environment
that has ``xrobotoolkit_sdk`` installed (for example GR00T's ``.venv_teleop``)
without importing Axol's full CLI dependency tree.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from almond_axol.pico_bridge import PicoAxolBridge, PicoBridgeConfig  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge PICO XRoboToolkit tracking to Axol's VR WebSocket server."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/ws")
    parser.add_argument("--no-tls", action="store_true")
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument("--frequency", type=float, default=90.0)
    parser.add_argument(
        "--ee-source",
        choices=("controller", "wrist", "hand"),
        default="controller",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("unity", "webxr", "axol-vr", "body"),
        default="unity",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("tracking", "controller", "identity"),
        default="tracking",
    )
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument(
        "--freshness-mode",
        choices=("timestamp", "local"),
        default="timestamp",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("invert-trigger", "direct-trigger", "constant-open"),
        default="invert-trigger",
    )
    parser.add_argument(
        "--elbow-source",
        choices=("body", "frozen", "synthetic"),
        default="body",
    )
    parser.add_argument("--body-forward-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--lock-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-deadzone", type=float, default=0.02)
    parser.add_argument("--grip-deadzone", type=float, default=0.02)
    parser.add_argument("--auto-engage", action="store_true")
    parser.add_argument("--stale-timeout-s", type=float, default=0.25)
    parser.add_argument("--wait-body-timeout-s", type=float, default=60.0)
    parser.add_argument("--service-script", default="/opt/apps/roboticsservice/runService.sh")
    parser.add_argument("--no-start-service", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), force=True)
    cfg = PicoBridgeConfig(
        host=args.host,
        port=args.port,
        path=args.path,
        tls=not args.no_tls,
        verify_tls=args.verify_tls,
        frequency=args.frequency,
        ee_source=args.ee_source,
        coordinate_mode=args.coordinate_mode,
        orientation_mode=args.orientation_mode,
        position_scale=args.position_scale,
        freshness_mode=args.freshness_mode,
        gripper_mode=args.gripper_mode,
        elbow_source=args.elbow_source,
        body_forward_sign=args.body_forward_sign,
        grip_deadzone=args.grip_deadzone,
        lock_threshold=args.lock_threshold,
        trigger_deadzone=args.trigger_deadzone,
        auto_engage=args.auto_engage,
        stale_timeout_s=args.stale_timeout_s,
        wait_body_timeout_s=args.wait_body_timeout_s,
        service_script=args.service_script,
        start_service=not args.no_start_service,
        dry_run=args.dry_run,
    )
    asyncio.run(PicoAxolBridge(cfg).run())


if __name__ == "__main__":
    main()
