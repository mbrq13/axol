"""axol pico-bridge

Bridge PICO XRoboToolkit tracking into Axol's VR WebSocket protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import logging


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pico-bridge",
        help="Bridge PICO XRoboToolkit tracking to Axol VR teleop.",
        description=(
            "Reads PICO XRoboToolkit controller/body-tracking data and sends "
            "Axol-compatible VRFrame JSON to an already-running axol teleop or "
            "collect-data VRServer."
        ),
    )
    parser.add_argument("--host", default="localhost", help="Axol VRServer host.")
    parser.add_argument("--port", type=int, default=8000, help="Axol VRServer port.")
    parser.add_argument("--path", default="/ws", help="Axol VRServer WebSocket path.")
    parser.add_argument(
        "--no-tls",
        action="store_true",
        help="Use ws:// instead of wss://. Axol VRServer defaults to WSS.",
    )
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        help="Verify the server TLS certificate. Off by default for Axol's self-signed cert.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=90.0,
        help="Bridge send frequency in Hz.",
    )
    parser.add_argument(
        "--ee-source",
        choices=("controller", "wrist", "hand"),
        default="controller",
        help=(
            "Source for l_ee/r_ee. controller matches Axol's current WebXR "
            "semantics; wrist/hand use XRoboToolkit body joints."
        ),
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("unity", "webxr", "axol-vr", "body"),
        default="unity",
        help=(
            "Input coordinate convention. unity converts XRoboToolkit's Unity-like "
            "X-right/Y-up/Z-forward poses into Axol VRFrame coordinates. webxr "
            "uses X-right/Y-up/Z-backward. body calibrates from pelvis/neck/"
            "shoulders. axol-vr sends values through unchanged."
        ),
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("tracking", "controller", "identity"),
        default="tracking",
        help=(
            "Use tracked rotations, controller rotations with body wrist/hand "
            "positions, or identity rotations for diagnosis."
        ),
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help="Scale PICO positions before sending VRFrame. Use <1.0 if motion is too large.",
    )
    parser.add_argument(
        "--freshness-mode",
        choices=("timestamp", "local"),
        default="timestamp",
        help=(
            "How to decide tracking freshness. timestamp is safer; local treats "
            "each successful body read as fresh for SDKs with frozen timestamps."
        ),
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("invert-trigger", "direct-trigger", "constant-open"),
        default="invert-trigger",
        help="Map trigger input to Axol gripper command.",
    )
    parser.add_argument(
        "--elbow-source",
        choices=("body", "frozen", "synthetic"),
        default="body",
        help=(
            "Use live body elbows, freeze elbows at first sample, or synthesize "
            "elbows from shoulder-to-hand/controller geometry."
        ),
    )
    parser.add_argument(
        "--body-forward-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Only for --coordinate-mode body. Flip calibrated forward/backward axis.",
    )
    parser.add_argument(
        "--lock-threshold",
        type=float,
        default=0.5,
        help="Analog grip threshold used for l_lock/r_lock.",
    )
    parser.add_argument(
        "--trigger-deadzone",
        type=float,
        default=0.02,
        help="Ignore trigger values below this analog threshold.",
    )
    parser.add_argument(
        "--grip-deadzone",
        type=float,
        default=0.02,
        help="Ignore grip values below this analog threshold.",
    )
    parser.add_argument(
        "--auto-engage",
        action="store_true",
        help=(
            "Send both locks=true whenever tracking is fresh. Useful when "
            "XRoboToolkit streams poses but controller grip/button inputs stay at zero."
        ),
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=0.25,
        help="If body tracking is older than this, send locks=false.",
    )
    parser.add_argument(
        "--wait-body-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for XRoboToolkit body tracking before failing. 0 waits forever.",
    )
    parser.add_argument(
        "--service-script",
        default="/opt/apps/roboticsservice/runService.sh",
        help="XRoboToolkit service script to start before xrt.init().",
    )
    parser.add_argument(
        "--no-start-service",
        action="store_true",
        help="Do not launch the XRoboToolkit service script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not connect to Axol; print/log generated frames for diagnostics.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    from ..pico_bridge import PicoAxolBridge, PicoBridgeConfig

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
