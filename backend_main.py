import argparse
import asyncio
import json
import sys
from pathlib import Path

from common.structured_log import slog

from config import BackendConfig
from core import EventBus


def load_dope_model():
    return None  # TODO: initialized in main() via CLI


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _prompt_bool(question: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{question} [{suffix}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


def _prompt_str(question: str, default: str) -> str:
    raw = input(f"{question} [{default}]: ").strip()
    return default if raw == "" else raw


def _prompt_int(question: str, default: int) -> int:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            continue


def _parse_key_float_map(raw: str | None) -> dict[str, float]:
    """Parse 'k1:v1,k2:v2' into dict[str, float]."""
    out: dict[str, float] = {}
    if raw is None:
        return out
    s = str(raw).strip()
    if not s:
        return out
    for token in s.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        k, v = token.split(":", 1)
        k = k.strip()
        if not k:
            continue
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    return out


def _collect_cli_flags(argv: list[str] | None) -> set[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    provided: set[str] = set()
    for token in raw_args:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        if key.startswith("no-"):
            key = key[3:]
        provided.add(key)
    return provided


def _bus_pending_count(bus: EventBus) -> int:
    total = 0
    for queues in bus._queues.values():  # type: ignore[attr-defined]
        for q in queues:
            total += int(q.qsize())
    return total


def _load_profile_data(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Profile file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML profile requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError("Profile root must be a mapping/object")
    return data


def _arg_key_from_attr(attr: str) -> str:
    return attr.replace("_", "-")


def _apply_profile(args: argparse.Namespace, cli_flags: set[str]) -> None:
    if not getattr(args, "profile", None):
        return

    profile = _load_profile_data(str(args.profile))
    alias_map = {
        "source": "input_mode",
        "session": "input_session",
        "loop": "replay_loop",
        "fps": "replay_fps",
    }

    for raw_key, value in profile.items():
        key = str(raw_key).strip().replace("-", "_")
        target = alias_map.get(key, key)
        if not hasattr(args, target):
            continue

        arg_key = _arg_key_from_attr(target)
        if arg_key in cli_flags:
            continue

        setattr(args, target, value)

    if getattr(args, "source", None) is not None and "source" in cli_flags:
        args.input_mode = args.source
    if getattr(args, "session", None) is not None and "session" in cli_flags:
        args.input_session = args.session
    if getattr(args, "loop", None) is not None and "loop" in cli_flags:
        args.replay_loop = args.loop
    if getattr(args, "fps", None) is not None and "fps" in cli_flags:
        args.replay_fps = args.fps


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR/AR navigation backend")

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional run profile (.yaml/.yml/.json) with argument-style keys",
    )

    parser.add_argument("--prompt", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--sensor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--calibration", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--frontend", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--foundationpose", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--baseline", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--navigation", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--unity-bridge", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record-out-dir", type=str, default=None)
    parser.add_argument(
        "--record-run-name",
        type=str,
        default=None,
        help=(
            "Optional new subdirectory under --record-out-dir; default uses "
            "the next increasing results_N name"
        ),
    )
    parser.add_argument(
        "--record-vis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record shared visualization frames (YOLO/DOPE/AprilTag)",
    )
    parser.add_argument(
        "--record-vis-fps",
        type=float,
        default=None,
        help="FPS for recorded visualization videos (default: 30)",
    )
    parser.add_argument(
        "--record-frontend-vis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy alias of --record-vis",
    )
    parser.add_argument(
        "--record-frontend-vis-fps",
        type=float,
        default=None,
        help="Legacy alias of --record-vis-fps",
    )

    # Input source
    parser.add_argument(
        "--input-mode",
        type=str,
        default="live",
        choices=["live", "replay"],
        help="Input source mode: live devices or replay from session folder",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["live", "replay"],
        help="Alias of --input-mode",
    )
    parser.add_argument(
        "--input-session",
        type=str,
        default=None,
        help="Replay session folder under inputs/<timestamp>",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Alias of --input-session",
    )
    parser.add_argument(
        "--replay-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Loop replay H5 streams when reaching EOF",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Alias of --replay-loop",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=None,
        help="Replay publish FPS override (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Alias of --replay-fps",
    )

    # SensorManager (sub-sensors)
    parser.add_argument("--hl2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hl2-host", type=str, default=None)
    parser.add_argument("--hl2-calibration-path", type=str, default=None)
    parser.add_argument(
        "--hl2-ahat", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--hl2-pv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--hl2-imu", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument(
        "--realsense", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--realsense-show", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--realsense-serial",
        type=str,
        default=None,
        help="Deprecated: use --realsense-serials",
    )
    parser.add_argument("--realsense-serials", nargs="+", type=str, default=None)
    parser.add_argument("--realsense-width", type=int, default=None)
    parser.add_argument("--realsense-height", type=int, default=None)
    parser.add_argument("--realsense-fps", type=int, default=None)
    parser.add_argument(
        "--realsense-color", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--realsense-depth", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--quest", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quest-host", type=str, default=None)
    parser.add_argument("--quest-port", type=int, default=None)
    parser.add_argument("--quest-world-extrinsic", type=str, default=None)
    parser.add_argument("--quest-init-output", type=str, default=None)
    parser.add_argument("--quest-init-reference-extrinsics", type=str, default=None)
    parser.add_argument("--quest-init-min-samples", type=int, default=None)
    parser.add_argument(
        "--quest-require-sensor-command",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # TrackingManager
    parser.add_argument(
        "--tracking-use-realsense", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-use-hl2-pv", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-use-quest", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-ahat-markers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-apriltag", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-apriltag-show", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--tracking-dope-show", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--tracking-apriltag-tag-size-mm", type=float, default=None)
    parser.add_argument(
        "--tracking-apriltag-family",
        type=str,
        default=None,
        help="Comma-separated tag families, e.g. TAG16H5 (default: TAG16H5)",
    )
    parser.add_argument(
        "--cube-model",
        type=str,
        default=None,
        help="Path to calibrated cube_model.yaml. Enables cube-model-based tag-cube "
        "pose estimation (no hard-coded face mapping / corner-roll search).",
    )

    # DOPE
    parser.add_argument("--dope-root", type=str, default=None)
    parser.add_argument("--dope-weights", type=str, default=None)
    parser.add_argument("--dope-object", type=str, default=None)
    parser.add_argument("--dope-config", type=str, default=None)
    parser.add_argument(
        "--dope-parallel", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dope-device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--dope-downscale-height", type=int, default=0,
        help="Resize input to this height before DOPE inference (0 = no downscale, best quality)")

    # Frontend (segmentation-based observations)
    parser.add_argument("--frontend-detector", type=str, default=None)
    parser.add_argument("--frontend-model", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--frontend-target-classes", type=str, default=None)

    # FoundationPose + HOPE CAD tracking
    parser.add_argument("--foundationpose-root", type=str, default=None)
    parser.add_argument("--hope-models-root", type=str, default=None)
    parser.add_argument(
        "--foundationpose-objects",
        type=str,
        default=None,
        help="Comma-separated HOPE object names, e.g. TomatoSauce,Tuna",
    )
    parser.add_argument("--foundationpose-source", type=str, default=None)
    parser.add_argument(
        "--foundationpose-class-map",
        type=str,
        default=None,
        help="YOLO-to-HOPE names, e.g. can:TomatoSauce,tuna:Tuna",
    )
    parser.add_argument("--foundationpose-initial-mask", type=str, default=None)
    parser.add_argument("--foundationpose-register-iterations", type=int, default=None)
    parser.add_argument("--foundationpose-track-iterations", type=int, default=None)
    parser.add_argument(
        "--foundationpose-score-tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export FoundationPose rank diagnostics during tracking (not a calibrated probability)",
    )
    parser.add_argument(
        "--foundationpose-depth-fit-quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render CAD depth for an optional q_fit quality term; adds a GPU render pass",
    )
    parser.add_argument("--foundationpose-depth-fit-interval", type=float, default=None)
    parser.add_argument(
        "--foundationpose-recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CNOS CAD masks to periodically re-register after occlusion",
    )
    parser.add_argument("--foundationpose-recovery-interval", type=float, default=None)
    parser.add_argument("--foundationpose-recovery-confidence", type=float, default=None)
    parser.add_argument(
        "--foundationpose-recovery-confirmations", type=int, default=None
    )
    parser.add_argument(
        "--foundationpose-recovery-misses-before-lost", type=int, default=None
    )
    parser.add_argument(
        "--foundationpose-recovery-min-score-margin", type=float, default=None
    )
    parser.add_argument("--cnos-root", type=str, default=None)
    parser.add_argument("--cnos-template-dir", type=str, default=None)
    parser.add_argument(
        "--cnos-sam-model", choices=["vit_b", "vit_l", "vit_h"], default=None
    )
    parser.add_argument("--cnos-sam-checkpoint", type=str, default=None)
    parser.add_argument("--cnos-dino-model", type=str, default=None)
    parser.add_argument("--cnos-dino-root", type=str, default=None)
    parser.add_argument("--cnos-dino-checkpoint", type=str, default=None)
    parser.add_argument("--cnos-device", type=str, default=None)
    parser.add_argument("--cnos-image-width", type=int, default=None)

    # Baseline controls
    parser.add_argument("--baseline-single-camera", type=str, default=None)

    # Fusion ablations
    parser.add_argument(
        "--fusion-adaptive-cov",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-online-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-integrity-monitoring",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-temporal-factors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-time-compensation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-mahalanobis-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fusion-sensor-fault-isolation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fusion-covariance-calibration", type=str, default=None)

    # Stress experiment controls
    parser.add_argument(
        "--exp-mode",
        type=str,
        default=None,
        help="Deprecated marker only; use explicit exp-* controls",
    )
    parser.add_argument(
        "--exp-delay",
        type=str,
        default=None,
        help="Per-camera delay map: cam1:0.05,cam2:0.02",
    )
    parser.add_argument(
        "--exp-drop",
        type=str,
        default=None,
        help="Per-camera drop prob map: cam1:0.2,cam2:0.1",
    )
    parser.add_argument("--exp-noise-trans", type=float, default=None)
    parser.add_argument("--exp-noise-rot", type=float, default=None)
    parser.add_argument(
        "--exp-camera-start",
        type=str,
        default=None,
        help="Per-camera activation time map (s): cam3:10.0",
    )

    return parser.parse_args(argv)


async def main(argv: list[str] | None = None):
    cli_flags = _collect_cli_flags(argv)
    args = _parse_args(argv)
    _apply_profile(args, cli_flags)

    cfg = BackendConfig()

    # CLI overrides (highest priority)
    if args.hl2_host is not None:
        cfg.hl2.host = args.hl2_host
    if args.hl2_calibration_path is not None:
        cfg.hl2.calibration_path = args.hl2_calibration_path

    if args.realsense_serial is not None:
        cfg.camera.serial = args.realsense_serial
    if args.realsense_width is not None:
        cfg.camera.width = args.realsense_width
    if args.realsense_height is not None:
        cfg.camera.height = args.realsense_height
    if args.realsense_fps is not None:
        cfg.camera.fps = args.realsense_fps
    if args.realsense_color is not None:
        cfg.camera.use_color = args.realsense_color
    if args.realsense_depth is not None:
        cfg.camera.use_depth = args.realsense_depth

    if args.frontend_detector is not None:
        cfg.frontend.detector = str(args.frontend_detector)
    if args.frontend_model is not None:
        cfg.frontend.model_path = str(args.frontend_model)
    if args.frontend_device is not None:
        cfg.frontend.device = str(args.frontend_device)
    if args.frontend_target_classes is not None:
        cfg.frontend.target_classes = [
            c.strip() for c in str(args.frontend_target_classes).split(",") if c.strip()
        ]
    if args.foundationpose_root is not None:
        cfg.foundationpose.foundationpose_root = str(args.foundationpose_root)
    if args.hope_models_root is not None:
        cfg.foundationpose.hope_models_root = str(args.hope_models_root)
    if args.foundationpose_objects is not None:
        cfg.foundationpose.objects = [
            item.strip()
            for item in str(args.foundationpose_objects).split(",")
            if item.strip()
        ]
    if args.foundationpose_source is not None:
        cfg.foundationpose.source = str(args.foundationpose_source)
    if args.foundationpose_class_map is not None:
        cfg.foundationpose.class_to_object = {}
        for entry in str(args.foundationpose_class_map).split(","):
            class_name, separator, object_name = entry.partition(":")
            if not separator or not class_name.strip() or not object_name.strip():
                raise ValueError(
                    "--foundationpose-class-map entries must be class:HOPEObject"
                )
            cfg.foundationpose.class_to_object[class_name.strip()] = object_name.strip()
    if args.foundationpose_initial_mask is not None:
        cfg.foundationpose.initial_mask_path = str(args.foundationpose_initial_mask)
    if args.foundationpose_register_iterations is not None:
        cfg.foundationpose.register_iterations = int(
            args.foundationpose_register_iterations
        )
    if args.foundationpose_track_iterations is not None:
        cfg.foundationpose.track_iterations = int(args.foundationpose_track_iterations)
    if args.foundationpose_score_tracking is not None:
        cfg.foundationpose.score_tracking = bool(args.foundationpose_score_tracking)
    if args.foundationpose_depth_fit_quality is not None:
        cfg.foundationpose.depth_fit_quality_enable = bool(args.foundationpose_depth_fit_quality)
    if args.foundationpose_depth_fit_interval is not None:
        cfg.foundationpose.depth_fit_quality_interval_s = max(
            0.0, float(args.foundationpose_depth_fit_interval)
        )
    if args.foundationpose_recovery is not None:
        cfg.foundationpose.recovery_enable = bool(args.foundationpose_recovery)
    if args.foundationpose_recovery_interval is not None:
        cfg.foundationpose.recovery_interval_s = max(
            0.0, float(args.foundationpose_recovery_interval)
        )
    if args.foundationpose_recovery_confidence is not None:
        cfg.foundationpose.recovery_conf_threshold = float(
            args.foundationpose_recovery_confidence
        )
    if args.foundationpose_recovery_confirmations is not None:
        cfg.foundationpose.recovery_confirmations = max(
            1, int(args.foundationpose_recovery_confirmations)
        )
    if args.foundationpose_recovery_misses_before_lost is not None:
        cfg.foundationpose.recovery_misses_before_lost = max(
            1, int(args.foundationpose_recovery_misses_before_lost)
        )
    if args.foundationpose_recovery_min_score_margin is not None:
        cfg.foundationpose.recovery_min_score_margin = max(
            0.0, float(args.foundationpose_recovery_min_score_margin)
        )
    if args.cnos_root is not None:
        cfg.foundationpose.cnos_root = str(args.cnos_root)
    if args.cnos_template_dir is not None:
        cfg.foundationpose.cnos_template_dir = str(args.cnos_template_dir)
    if args.cnos_sam_model is not None:
        cfg.foundationpose.cnos_sam_model_type = str(args.cnos_sam_model)
    if args.cnos_sam_checkpoint is not None:
        cfg.foundationpose.cnos_sam_checkpoint = str(args.cnos_sam_checkpoint)
    if args.cnos_dino_model is not None:
        cfg.foundationpose.cnos_dino_model_name = str(args.cnos_dino_model)
    if args.cnos_dino_root is not None:
        cfg.foundationpose.cnos_dino_root = str(args.cnos_dino_root)
    if args.cnos_dino_checkpoint is not None:
        cfg.foundationpose.cnos_dino_checkpoint = str(args.cnos_dino_checkpoint)
    if args.cnos_device is not None:
        cfg.foundationpose.cnos_device = str(args.cnos_device)
    if args.cnos_image_width is not None:
        cfg.foundationpose.cnos_image_width = max(0, int(args.cnos_image_width))

    if args.baseline_single_camera is not None:
        cfg.baseline.single_camera_source = str(args.baseline_single_camera)

    if args.fusion_adaptive_cov is not None:
        cfg.fusion.use_adaptive_covariance = bool(args.fusion_adaptive_cov)
    if args.fusion_online_calibration is not None:
        cfg.fusion.enable_online_calibration = bool(args.fusion_online_calibration)
    if args.fusion_integrity_monitoring is not None:
        cfg.fusion.enable_integrity_monitoring = bool(args.fusion_integrity_monitoring)
    if args.fusion_temporal_factors is not None:
        cfg.fusion.enable_temporal_factors = bool(args.fusion_temporal_factors)
    if args.fusion_time_compensation is not None:
        cfg.fusion.enable_time_compensation = bool(args.fusion_time_compensation)
    if args.fusion_mahalanobis_gate is not None:
        cfg.fusion.enable_mahalanobis_gate = bool(args.fusion_mahalanobis_gate)
    if args.fusion_sensor_fault_isolation is not None:
        cfg.fusion.enable_sensor_fault_isolation = bool(args.fusion_sensor_fault_isolation)
    if args.fusion_covariance_calibration is not None:
        cfg.fusion.covariance_calibration_path = str(args.fusion_covariance_calibration)

    if args.exp_mode is not None:
        slog(
            component="backend_main",
            event="deprecated_arg",
            arg="exp-mode",
            replacement=(
                "exp-delay/exp-drop/exp-noise-trans/exp-noise-rot/exp-camera-start"
            ),
        )
        cfg.experiment.mode = str(args.exp_mode)
    if args.exp_delay is not None:
        cfg.experiment.delay_s_by_camera = _parse_key_float_map(args.exp_delay)
    if args.exp_drop is not None:
        cfg.experiment.drop_prob_by_camera = _parse_key_float_map(args.exp_drop)
    if args.exp_noise_trans is not None:
        cfg.experiment.noise_trans_m = float(args.exp_noise_trans)
    if args.exp_noise_rot is not None:
        cfg.experiment.noise_rot_deg = float(args.exp_noise_rot)
    if args.exp_camera_start is not None:
        cfg.experiment.camera_start_time_s = _parse_key_float_map(args.exp_camera_start)

    bus = EventBus()
    loop = asyncio.get_running_loop()

    prompt_enabled = args.prompt if args.prompt is not None else _is_interactive()

    input_mode = str(args.input_mode or "live").strip().lower()
    replay_mode = input_mode == "replay"
    replay_session_dir: Path | None = None
    if replay_mode:
        if not args.input_session:
            raise ValueError(
                "Replay mode requires --input-session <inputs/<timestamp>>"
            )
        replay_session_dir = Path(args.input_session)
        if not replay_session_dir.exists() or not replay_session_dir.is_dir():
            raise FileNotFoundError(f"Replay session not found: {replay_session_dir}")

    enable_sensor = True if args.sensor is None else args.sensor
    enable_calibration = False if args.calibration is None else args.calibration
    enable_tracking = False if args.tracking is None else args.tracking
    enable_frontend = False if args.frontend is None else bool(args.frontend)
    enable_foundationpose = (
        False if args.foundationpose is None else bool(args.foundationpose)
    )
    enable_fusion = False if args.fusion is None else args.fusion
    enable_baseline = enable_tracking if args.baseline is None else bool(args.baseline)
    enable_navigation = False if args.navigation is None else args.navigation
    enable_unity_bridge = False if args.unity_bridge is None else args.unity_bridge
    enable_record = False if args.record is None else args.record
    if args.record_vis is None:
        record_vis = (
            False if args.record_frontend_vis is None else bool(args.record_frontend_vis)
        )
    else:
        record_vis = bool(args.record_vis)
    if args.record_vis_fps is not None:
        record_vis_fps = float(args.record_vis_fps)
    elif args.record_frontend_vis_fps is not None:
        record_vis_fps = float(args.record_frontend_vis_fps)
    else:
        record_vis_fps = 30.0

    enable_hl2 = True if args.hl2 is None else args.hl2
    enable_realsense = True if args.realsense is None else args.realsense
    enable_quest = False if args.quest is None else bool(args.quest)
    if args.quest_host is not None:
        cfg.quest.host = str(args.quest_host)
    if args.quest_port is not None:
        cfg.quest.port = int(args.quest_port)
    if args.quest_world_extrinsic is not None:
        cfg.quest.world_extrinsic_path = str(args.quest_world_extrinsic)
    if args.quest_init_output is not None:
        cfg.quest.init_output_path = str(args.quest_init_output)
    if args.quest_init_reference_extrinsics is not None:
        cfg.quest.init_reference_extrinsics_path = str(
            args.quest_init_reference_extrinsics
        )
    if args.quest_init_min_samples is not None:
        cfg.quest.init_min_samples = max(3, int(args.quest_init_min_samples))
    cfg.quest.init_cube_model_path = str(
        args.cube_model or cfg.calibration.tag_cube.cube_model_path or ""
    )
    if args.quest_require_sensor_command is not None:
        cfg.quest.require_sensor_command = bool(args.quest_require_sensor_command)
    realsense_show_preview = (
        False if args.realsense_show is None else args.realsense_show
    )
    hl2_enable_ahat = True if args.hl2_ahat is None else args.hl2_ahat
    hl2_enable_pv = True if args.hl2_pv is None else args.hl2_pv
    hl2_enable_imu = True if args.hl2_imu is None else args.hl2_imu

    tracking_use_realsense = (
        True if args.tracking_use_realsense is None else args.tracking_use_realsense
    )

    if enable_frontend and not enable_tracking:
        # In frontend-only mode, allow baseline to consume seg3d observations.
        cfg.baseline.dope_only = False
    tracking_use_hl2_pv = (
        False if args.tracking_use_hl2_pv is None else args.tracking_use_hl2_pv
    )
    tracking_use_quest = (
        True if args.tracking_use_quest is None else args.tracking_use_quest
    )
    tracking_ahat_markers = (
        True if args.tracking_ahat_markers is None else args.tracking_ahat_markers
    )

    tracking_apriltag = (
        True if args.tracking_apriltag is None else args.tracking_apriltag
    )
    tracking_apriltag_show = (
        False if args.tracking_apriltag_show is None else args.tracking_apriltag_show
    )
    tracking_dope_show = (
        False if args.tracking_dope_show is None else args.tracking_dope_show
    )
    # Default tag size: 40mm
    tracking_apriltag_tag_size_m = 0.04
    if args.tracking_apriltag_tag_size_mm is not None:
        tracking_apriltag_tag_size_m = (
            float(args.tracking_apriltag_tag_size_mm) / 1000.0
        )
    # Default tag family: TAG16H5 only
    _default_families = ["TAG16H5"]
    if args.tracking_apriltag_family:
        tracking_apriltag_families = [
            f.strip() for f in args.tracking_apriltag_family.split(",") if f.strip()
        ]
    else:
        tracking_apriltag_families = list(_default_families)

    # Build RealSense camera list from config, optionally overridden by CLI.
    realsense_cameras = list(getattr(cfg, "realsense_cameras", []) or [])
    realsense_serials_cli: list[str] = []
    if args.realsense_serials:
        realsense_serials_cli.extend([str(s) for s in args.realsense_serials if s])
    if args.realsense_serial:
        if realsense_serials_cli:
            slog(
                component="backend_main",
                event="deprecated_arg",
                arg="realsense-serial",
                replacement="realsense-serials",
            )
        realsense_serials_cli.append(str(args.realsense_serial))
    # de-duplicate while preserving order
    if realsense_serials_cli:
        realsense_serials_cli = list(dict.fromkeys(realsense_serials_cli))

    from dataclasses import replace

    if replay_mode and replay_session_dir is not None:
        replay_rgbd_root = replay_session_dir / "rgbd"
        replay_h5_serials: list[str] = []
        if replay_rgbd_root.exists() and replay_rgbd_root.is_dir():
            for p in sorted(replay_rgbd_root.glob("*.h5")):
                replay_h5_serials.append(p.stem)

        replay_serials = replay_h5_serials

        # --- Quest H5 discovery ---
        replay_quest_root = replay_session_dir / "quest"
        quest_topics: list[str] = []
        if replay_quest_root.exists() and replay_quest_root.is_dir():
            for p in sorted(replay_quest_root.glob("*.h5")):
                quest_name = f"quest/{p.stem}"
                quest_topics.append(f"{quest_name}/stream")

        if not replay_serials and not quest_topics:
            raise RuntimeError(
                f"No replay h5 files found in session/rgbd or session/quest: {replay_session_dir}"
            )

        if replay_serials:
            realsense_cameras = [
                replace(cfg.camera, name=f"replay_{i}", serial=s)
                for i, s in enumerate(replay_serials)
            ]
        else:
            realsense_cameras = []

        # Auto-load session extrinsics if available; otherwise clear (no fabricated extrinsics).
        extr_path = replay_session_dir / "tag_cube_extrinsics.json"
        if extr_path.exists():
            cfg.calibration.tag_cube.result_path = str(extr_path)
        else:
            cfg.calibration.tag_cube.result_path = ""
        for quest_extrinsic_name in ("quest_world_extrinsic.json", "tag_cube_pose.json"):
            quest_extrinsic_path = replay_session_dir / quest_extrinsic_name
            if quest_extrinsic_path.exists():
                cfg.quest.world_extrinsic_path = str(quest_extrinsic_path)
                break

    elif realsense_serials_cli:
        realsense_cameras = [
            replace(cfg.camera, name=f"realsense_{i}", serial=s)
            for i, s in enumerate(realsense_serials_cli)
        ]
        quest_topics = [f"{cfg.quest.source_name}/stream"] if enable_quest else []

    else:
        quest_topics = [f"{cfg.quest.source_name}/stream"] if enable_quest else []

    cfg.realsense_cameras = list(realsense_cameras)

    if prompt_enabled:
        enable_sensor = _prompt_bool("Enable SensorManager?", enable_sensor)
        enable_calibration = _prompt_bool(
            "Enable CalibrationManager?", enable_calibration
        )
        enable_tracking = _prompt_bool("Enable TrackingManager?", enable_tracking)
        enable_frontend = _prompt_bool(
            "Enable FrontendObservationManager?", enable_frontend
        )
        enable_fusion = _prompt_bool("Enable FusionManager?", enable_fusion)
        if enable_tracking and args.baseline is None:
            enable_baseline = _prompt_bool("Enable BaselineManager?", enable_baseline)
        enable_navigation = _prompt_bool("Enable NavigationManager?", enable_navigation)
        enable_unity_bridge = _prompt_bool(
            "Enable UnityBridgeManager?", enable_unity_bridge
        )
        enable_record = _prompt_bool("Enable RecordManager?", enable_record)

        if enable_sensor and not replay_mode:
            enable_hl2 = _prompt_bool("Enable HL2 streams?", enable_hl2)
            enable_realsense = _prompt_bool(
                "Enable RealSense streams?", enable_realsense
            )

            if enable_hl2:
                if args.hl2_host is None:
                    cfg.hl2.host = _prompt_str("HL2 host", cfg.hl2.host)
                if args.hl2_calibration_path is None:
                    cfg.hl2.calibration_path = _prompt_str(
                        "HL2 calibration path", cfg.hl2.calibration_path
                    )
                if args.hl2_ahat is None:
                    hl2_enable_ahat = _prompt_bool("HL2 enable AHAT?", hl2_enable_ahat)
                if args.hl2_pv is None:
                    hl2_enable_pv = _prompt_bool("HL2 enable PV?", hl2_enable_pv)
                if args.hl2_imu is None:
                    hl2_enable_imu = _prompt_bool("HL2 enable IMU?", hl2_enable_imu)

            if enable_realsense:
                if args.realsense_show is None:
                    realsense_show_preview = _prompt_bool(
                        "RealSense: show preview window?", realsense_show_preview
                    )
                if args.realsense_serial is None:
                    cfg.camera.serial = _prompt_str(
                        "RealSense serial (empty = auto)", cfg.camera.serial
                    )
                if args.realsense_width is None:
                    cfg.camera.width = _prompt_int("RealSense width", cfg.camera.width)
                if args.realsense_height is None:
                    cfg.camera.height = _prompt_int(
                        "RealSense height", cfg.camera.height
                    )
                if args.realsense_fps is None:
                    cfg.camera.fps = _prompt_int("RealSense fps", cfg.camera.fps)
                if args.realsense_color is None:
                    cfg.camera.use_color = _prompt_bool(
                        "RealSense enable color?", cfg.camera.use_color
                    )
                if args.realsense_depth is None:
                    cfg.camera.use_depth = _prompt_bool(
                        "RealSense enable depth?", cfg.camera.use_depth
                    )

        if enable_tracking:
            if args.tracking_use_realsense is None:
                tracking_use_realsense = _prompt_bool(
                    "Tracking: use RealSense stream?", tracking_use_realsense
                )
            if args.tracking_use_hl2_pv is None:
                tracking_use_hl2_pv = _prompt_bool(
                    "Tracking: use HL2 PV stream?", tracking_use_hl2_pv
                )
            if args.tracking_ahat_markers is None:
                tracking_ahat_markers = _prompt_bool(
                    "Tracking: enable HL2 AHAT markers?", tracking_ahat_markers
                )

            if args.tracking_apriltag is None:
                tracking_apriltag = _prompt_bool(
                    "Tracking: enable RealSense AprilTag detection?", tracking_apriltag
                )
            if args.tracking_apriltag_show is None:
                tracking_apriltag_show = _prompt_bool(
                    "Tracking: show ArUco overlay window?", tracking_apriltag_show
                )
            if args.tracking_dope_show is None:
                tracking_dope_show = _prompt_bool(
                    "Tracking: show DOPE overlay window?", tracking_dope_show
                )
            if args.tracking_apriltag_tag_size_mm is None:
                tracking_apriltag_tag_size_m = (
                    float(
                        _prompt_int(
                            "Tracking: ArUcoTag size (mm)",
                            int(tracking_apriltag_tag_size_m * 1000.0),
                        )
                    )
                    / 1000.0
                )
            if args.tracking_apriltag_family is None:
                _fam_input = _prompt_str(
                    "Tracking: tag families (comma-sep, e.g. TAG16H5,4X4_50)",
                    ",".join(tracking_apriltag_families),
                )
                tracking_apriltag_families = [
                    f.strip() for f in _fam_input.split(",") if f.strip()
                ]

    tasks: list[asyncio.Task] = []
    replay_task: asyncio.Task | None = None
    replay_mgr = None
    tracking_mgr = None
    frontend_mgr = None
    record_mgr = None

    foundationpose_mgr = None

    if enable_foundationpose and not cfg.foundationpose.objects:
        raise ValueError("--foundationpose requires --foundationpose-objects")

    if enable_baseline and not (
        enable_tracking or enable_frontend or enable_foundationpose
    ):
        slog(
            component="backend_main",
            event="baseline_disabled",
            reason="no_observation_source_enabled",
        )
        enable_baseline = False

    slog(
        component="backend_main",
        event="start",
        input_mode=input_mode,
        input_session=(None if replay_session_dir is None else str(replay_session_dir)),
        prompt_enabled=prompt_enabled,
        enable_sensor=enable_sensor,
        enable_calibration=enable_calibration,
        enable_tracking=enable_tracking,
        enable_frontend=enable_frontend,
        enable_foundationpose=enable_foundationpose,
        enable_fusion=enable_fusion,
        enable_baseline=enable_baseline,
        enable_navigation=enable_navigation,
        enable_unity_bridge=enable_unity_bridge,
        enable_hl2=enable_hl2,
        enable_realsense=enable_realsense,
        realsense_show_preview=realsense_show_preview,
        hl2_host=cfg.hl2.host,
        hl2_calibration_path=cfg.hl2.calibration_path,
        realsense_width=cfg.camera.width,
        realsense_height=cfg.camera.height,
        realsense_fps=cfg.camera.fps,
        realsense_serial=cfg.camera.serial,
        realsense_use_color=cfg.camera.use_color,
        realsense_use_depth=cfg.camera.use_depth,
        frontend_detector=cfg.frontend.detector,
        record_vis=(bool(enable_record) and bool(record_vis)),
        record_vis_fps=record_vis_fps,
        exp_mode=cfg.experiment.mode,
    )

    if replay_mode and replay_session_dir is not None:
        from managers.replay_input_manager import ReplayInputManager

        replay_mgr = ReplayInputManager(
            bus,
            cfg,
            session_dir=replay_session_dir,
            loop_playback=bool(args.replay_loop),
            fps_override=args.replay_fps,
        )
        replay_task = asyncio.create_task(replay_mgr.run())
        tasks.append(replay_task)
    elif enable_sensor:
        from managers.sensor_manager import SensorManager

        sensor_mgr = SensorManager(
            bus,
            loop,
            cfg,
            enable_hl2=enable_hl2,
            enable_realsense=enable_realsense,
            enable_quest=enable_quest,
            hl2_enable_ahat=hl2_enable_ahat,
            hl2_enable_pv=hl2_enable_pv,
            hl2_enable_imu=hl2_enable_imu,
            realsense_show_preview=realsense_show_preview,
            realsense_cameras=realsense_cameras,
        )
        sensor_mgr.start()

    if enable_calibration:
        from managers.calibration_manager import CalibrationManager

        calib_mgr = CalibrationManager(bus, cfg)
        tasks.append(asyncio.create_task(calib_mgr.run()))

    if enable_record:
        from managers.record_manager import RecordManager

        out_dir = args.record_out_dir or "records"
        record_mgr = RecordManager(
            bus,
            cfg,
            out_dir=out_dir,
            run_name=args.record_run_name,
            record_vis=bool(record_vis),
            vis_fps=record_vis_fps,
            vis_fps_by_source=(
                replay_mgr.recorded_fps_by_source
                if replay_mgr is not None
                and args.record_vis_fps is None
                and args.record_frontend_vis_fps is None
                else None
            ),
            vis_topic="visualization/frames",
            runtime_metadata={
                "input_mode": input_mode,
                "replay_session": str(args.input_session or ""),
                "artifact_paths": [args.dope_weights] if args.dope_weights else [],
            },
        )
        tasks.append(asyncio.create_task(record_mgr.run()))

    if enable_tracking:
        from managers.tracking_manager import TrackingManager

        dope_model = None
        if args.dope_weights and args.dope_object:
            from adapters.dope_runner import DopeRunner

            repo_root = Path(__file__).resolve().parent
            default_dope_root = (
                repo_root / ".." / "DOPE" / "Deep_Object_Pose"
            ).resolve()
            dope_root = args.dope_root or str(default_dope_root)
            dope_cfg = args.dope_config

            # Support comma-separated multiple objects: --dope-weights w1,w2,w3 --dope-object o1,o2,o3
            weight_list = [w.strip() for w in str(args.dope_weights).split(",") if w.strip()]
            object_list = [o.strip() for o in str(args.dope_object).split(",") if o.strip()]
            if not weight_list or not object_list:
                raise ValueError("--dope-weights and --dope-object must specify at least one object")
            if len(weight_list) != len(object_list):
                raise ValueError(
                    f"Mismatch: {len(weight_list)} weights but {len(object_list)} objects. "
                    "Each object needs one weights file."
                )

            dope_model = DopeRunner(
                dope_root=dope_root,
                objects=list(zip(object_list, weight_list)),
                config_path=dope_cfg,
                parallel=bool(args.dope_parallel),
                device=str(args.dope_device),
                downscale_height=args.dope_downscale_height,
            )

        realsense_topics = ["realsense_stream"]
        if realsense_cameras:
            realsense_topics = [
                f"realsense/{(c.serial or c.name or 'realsense')}/stream"
                for c in realsense_cameras
            ]

        tracking_mgr = TrackingManager(
            bus,
            loop,
            model=dope_model,
            use_realsense_for_dope=tracking_use_realsense,
            use_hl2_pv_for_dope=tracking_use_hl2_pv,
            use_quest_for_dope=tracking_use_quest,
            enable_ahat_markers=tracking_ahat_markers,
            enable_realsense_apriltag=tracking_apriltag,
            apriltag_tag_size_m=tracking_apriltag_tag_size_m,
            apriltag_families=tracking_apriltag_families,
            apriltag_show_preview=tracking_apriltag_show,
            dope_show_preview=tracking_dope_show,
            realsense_topics=realsense_topics,
            quest_topics=quest_topics,
            calibration_result_path=str(cfg.calibration.tag_cube.result_path),
            sensor_modes=cfg.calibration.per_camera_mode,
            quest_world_extrinsic_path=cfg.quest.world_extrinsic_path,
            quest_tag_correction_alpha=cfg.quest.tag_correction_alpha,
            quest_tag_correction_max_translation_m=cfg.quest.tag_correction_max_translation_m,
            quest_tag_correction_max_rotation_deg=cfg.quest.tag_correction_max_rotation_deg,
            cube_model_path=(
                args.cube_model
                or cfg.calibration.tag_cube.cube_model_path
                or None
            ),
        )
        if replay_mode:
            tracking_mgr.configure_runtime(drop_old_frames=False, dope_min_interval_s=0.0)
        tracking_mgr.set_visualization_stream(
            enabled=(bool(enable_record) and bool(record_vis)),
            topic="visualization/frames",
            publish_dope=True,
            publish_apriltag=True,
        )
        tasks.append(asyncio.create_task(tracking_mgr.run()))

    if enable_frontend:
        from managers.frontend_observation_manager import FrontendObservationManager

        cfg.frontend.enable = True
        frontend_mgr = FrontendObservationManager(bus, cfg, input_mode=input_mode)
        frontend_mgr.set_visualization_stream(
            enabled=(bool(enable_record) and bool(record_vis)),
            topic="visualization/frames",
            stream="all_vis",
        )
        tasks.append(asyncio.create_task(frontend_mgr.run()))

    if enable_foundationpose:
        from managers.foundationpose_manager import FoundationPoseManager

        foundationpose_topics = ["realsense_stream"]
        if realsense_cameras:
            foundationpose_topics = [
                f"realsense/{(camera.serial or camera.name or 'realsense')}/stream"
                for camera in realsense_cameras
            ]
        foundationpose_topics.extend(quest_topics)
        foundationpose_topics = list(dict.fromkeys(foundationpose_topics))
        foundationpose_mgr = FoundationPoseManager(
            bus,
            cfg.foundationpose,
            realsense_topics=foundationpose_topics,
            calibration_result_path=str(cfg.calibration.tag_cube.result_path),
            mobile_world_extrinsic_path=cfg.quest.world_extrinsic_path,
            input_mode=input_mode,
        )
        foundationpose_mgr.set_visualization_stream(
            enabled=(bool(enable_record) and bool(record_vis)),
            topic="visualization/frames",
        )
        tasks.append(asyncio.create_task(foundationpose_mgr.run()))

    if enable_baseline:
        from managers.baseline_manager import BaselineManager

        baseline_mgr = BaselineManager(bus, cfg)
        tasks.append(asyncio.create_task(baseline_mgr.run()))

    if enable_fusion:
        from managers.fusion_manager import FusionManager

        fusion_mgr = FusionManager(bus, loop, cfg)
        fusion_mgr.configure_runtime(offline_replay=bool(replay_mode))
        tasks.append(asyncio.create_task(fusion_mgr.run()))

    if enable_navigation:
        from managers.navigation_manager import NavigationManager

        nav_mgr = NavigationManager(bus)
        tasks.append(asyncio.create_task(nav_mgr.run()))

    if enable_unity_bridge:
        from managers.unity_bridge import UnityBridgeManager

        unity_bridge = UnityBridgeManager(bus)
        tasks.append(asyncio.create_task(unity_bridge.run()))

    if tasks:
        if replay_mode and replay_task is not None:
            other_tasks = [t for t in tasks if t is not replay_task]
            replay_exc: BaseException | None = None
            try:
                await replay_task
            except BaseException as exc:  # propagate after cleanup
                replay_exc = exc
            finally:
                while True:
                    pending = _bus_pending_count(bus)
                    if tracking_mgr is not None:
                        pending += int(tracking_mgr.pending_input_count())
                    if frontend_mgr is not None:
                        pending += int(frontend_mgr.pending_input_count())
                    if foundationpose_mgr is not None:
                        pending += int(foundationpose_mgr.pending_input_count())
                    if record_mgr is not None:
                        pending += int(record_mgr.pending_item_count())
                    if pending <= 0:
                        break
                    await asyncio.sleep(0.02)

                for t in other_tasks:
                    t.cancel()
                if other_tasks:
                    await asyncio.gather(*other_tasks, return_exceptions=True)
            if replay_exc is not None:
                raise replay_exc
        else:
            await asyncio.gather(*tasks)
    else:
        # keep alive for background sensor tasks
        await asyncio.Event().wait()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()  # 可选但推荐（兼容某些打包场景）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    asyncio.run(main())
