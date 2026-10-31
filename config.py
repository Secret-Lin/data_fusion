# config.py

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    name: str = "realsenseD435i"
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: str = ""  # 可选：指定某个RealSense
    use_depth: bool = True
    use_color: bool = True
    sync_mode: int = 0  # 0: Default, 1: Master, 2: Slave
    visual_preset: str = "High Accuracy"  # "High Accuracy", "High Density", "Default"
    apriltag_families: list[str] = field(default_factory=lambda: ["TAG16H5"])
    # 这里可以加 device_index / RTSP URL 等


@dataclass
class HL2Config:
    host: str = "127.0.0.1"
    port: int = 12345
    calibration_path: str = "calibration/hl2"
    # 具体根据 HL2SS 配置调整


@dataclass
class OTSConfig:
    server_ip: str = "127.0.0.1"
    server_port: int = 1510
    # 具体根据 NatNet 配置调整


@dataclass
class QuestConfig:
    host: str = "0.0.0.0"
    port: int = 5001
    source_name: str = "quest/quest"
    require_sensor_command: bool = True
    queue_size: int = 8
    depth_valid_min_m: float = 0.15
    depth_valid_max_m: float = 10.0
    alignment_splat_radius: int = 1
    fill_depth_holes: bool = False
    depth_hole_fill_kernel: int = 3
    alignment_rotate_180: bool = False
    alignment_flip_horizontal: bool = False
    alignment_flip_vertical: bool = False
    allow_legacy_resize_fallback: bool = False
    world_extrinsic_path: str = ""
    tag_correction_alpha: float = 0.2
    tag_correction_max_translation_m: float = 0.25
    tag_correction_max_rotation_deg: float = 30.0
    init_cube_model_path: str = "standalone/cube_model.yaml"
    init_reference_extrinsics_path: str = "calibration/tag_cube_extrinsics.json"
    init_min_samples: int = 20
    init_output_path: str = "quest_world_extrinsic.json"


# ---------------------------------------------------------------------------
# Tag-Cube Calibration
# ---------------------------------------------------------------------------


@dataclass
class TagCubeCalibrationConfig:
    # Cube geometry (mm)
    cube_size_mm: float = 50.0

    # Path to the calibrated cube model YAML (cube_model.yaml).
    # When set, tag-cube pose estimation uses the calibrated T_world_tag layout
    # (no hard-coded face mapping, no corner-roll search) instead of the legacy
    # hard-coded convention. Empty disables cube-model-based tracking.
    cube_model_path: str = "standalone/cube_model.yaml"

    # Tag family (all 6 faces use the same family)
    tag_family: str = "TAG16H5"

    # IDs on each face (see utils/tag_cube_calibrator.py for convention)
    # Default matches the standard convention:
    #   face +Z: id=0,  face +X: id=1,  face +Y: id=2
    #   face -X: id=3,  face -Y: id=4,  face -Z: id=5
    face_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Calibration collection settings
    min_frames: int = 20  # minimum frames per camera before solving
    min_visible_faces: int = 2  # minimum faces visible per frame

    # Output path for the tag-cube calibration result
    result_path: str = "calibration/tag_cube_extrinsics.json"


@dataclass
class CalibrationConfig:
    # Calibration mode for each camera
    # "fixed"   — use known extrinsics from tag-cube calibration
    # "unknown" — online self-calibration (extrinsic is a graph variable)
    # "mobile"  — camera is not stationary (e.g. handheld RealSense)
    per_camera_mode: dict[str, str] = field(default_factory=dict)

    # Path to tag-cube calibration result (JSON from tag_cube_calibrator.py)
    tag_cube_result_path: str = "calibration/tag_cube_extrinsics.json"

    tag_cube: TagCubeCalibrationConfig = field(default_factory=TagCubeCalibrationConfig)


@dataclass
class FusionConfig:
    # ---- topic names ----
    topic_hl2: str = "pose.hl2"
    topic_optitrack: str = "pose.optitrack"
    topic_fused: str = "pose.fused"

    # ---- fusion backend ----
    # Set to "gtsam" to use the new GTSAM iSAM2 HeteroFusionSystem.
    # Set to "ekf" to use the legacy EKF SE(3) fusion.
    backend: str = "gtsam"

    # ---- GTSAM iSAM2 parameters ----
    # Prior noise (initial tool pose uncertainty)
    prior_rot_sigma: float = 1e-2  # rad
    prior_trans_sigma: float = 1e-2  # m

    # Motion prior noise — tool (rigidly attached)
    motion_rot_sigma: float = 5e-2  # rad per step
    motion_trans_sigma: float = 5e-3  # m per step

    # Motion prior noise — mobile camera
    cam_motion_rot_sigma: float = 5e-2
    cam_motion_trans_sigma: float = 5e-3

    # Base observation uncertainty (before quality scaling)
    sigma_R: float = 1e-2  # rad — rotational
    sigma_t: float = 2e-3  # m — translational

    # Per-detector measurement uncertainty. DetectionResult.pose_world is
    # always a world-frame pose, irrespective of the originating camera.
    dope_rot_sigma: float = 0.14  # rad (~8 deg)
    dope_trans_sigma: float = 0.03  # m
    foundationpose_rot_sigma: float = 0.035  # rad (~2 deg)
    foundationpose_trans_sigma: float = 0.01  # m
    # Explicit uncalibrated prior for the M3T frontend; same starting scale as FP.
    m3t_rot_sigma: float = 0.035  # rad; not a learned/calibrated probability
    m3t_trans_sigma: float = 0.01  # m
    megapose_rot_sigma: float = 0.035  # rad; initial uncalibrated prior
    megapose_trans_sigma: float = 0.01  # m
    seg3d_rot_sigma: float = 0.35  # centroid PCA/identity orientation is weak
    seg3d_trans_sigma: float = 0.04  # m

    # Observations within this host-time window constrain the same graph state.
    sync_window_s: float = 0.04

    # Robust loss and same-state innovation gate.
    robust_huber_k: float = 1.345
    innovation_gate_trans_m: float = 0.15
    innovation_gate_rot_deg: float = 45.0
    # Use the statistically normalized innovation (NIS) when a covariance is
    # available.  The legacy geometric gate remains as a safety upper bound.
    enable_mahalanobis_gate: bool = True
    # Re-evaluate all observations of a state when another view arrives.
    # The legacy sequential hard gate is retained only for explicit ablations.
    enable_revisable_consensus: bool = True
    # For consecutive revisions of one world-pose state, solve from the same
    # pre-state iSAM2 checkpoint and the current consensus seed. This removes
    # the nonlinear initialization left by a discarded provisional sensor.
    # Opt-in pending held-out validation: order invariance alone does not
    # guarantee lower tracking error with imperfect observation covariances.
    enable_world_revision_checkpoint: bool = False
    consensus_use_nis: bool = False
    # Optional ambiguity safeguard: allow a much better RGB-D/CAD quality
    # singleton to beat a low-quality majority cluster.  Disabled until
    # calibrated on held-out data; the default preserves the established
    # cardinality-first consensus behavior.
    enable_consensus_quality_override: bool = False
    consensus_quality_override_ratio: float = 1.5
    nis_gate_probability: float = 0.99
    nis_gate_consecutive_failures: int = 2

    # Adaptive quality weights (for covariance scaling)
    w_area: float = 1.0  # detection area / feature count
    w_fit: float = 1.0  # reprojection / model fit
    w_time: float = 1.0  # temporal consistency
    covariance_min_scale: float = 0.25
    covariance_max_scale: float = 100.0
    covariance_floor_trans_m: float = 1e-4
    covariance_floor_rot_rad: float = 1e-4
    covariance_calibration_path: str = ""

    # Full temporal/time-alignment path.  These switches default to enabled;
    # disabling them reproduces the historical ablation behavior.
    enable_temporal_factors: bool = True
    # Use the previously fused object state as a tie-break/prediction when
    # revisable same-frame consensus selects among compatible hypotheses.  This
    # is intentionally separate from ``enable_temporal_factors``: disabling a
    # graph motion factor must not silently remove (or retain) the history
    # prior used by the consensus stage.
    enable_consensus_history_prediction: bool = True
    # Experimental safeguard: only recent, unambiguous multi-view states may
    # supply world-pose consensus prediction/motion. Live defaults unchanged.
    consensus_supported_history_only: bool = False
    # Offline candidate: brief prediction-only output when an unsupported
    # singleton jumps away from recent multi-view-confirmed history.
    enable_singleton_history_guard: bool = False
    singleton_history_guard_max_age_s: float = 0.15
    enable_time_compensation: bool = True
    max_reorder_s: float = 0.25
    max_prediction_gap_s: float = 0.5
    temporal_constant_velocity: bool = True
    temporal_process_scale: float = 1.0

    # Online calibration and sensor fault handling.
    online_extrinsic_prior_rot_sigma: float = 0.35
    online_extrinsic_prior_trans_sigma: float = 0.10
    online_extrinsic_max_correction_m: float = 0.25
    online_extrinsic_max_correction_deg: float = 30.0
    # Health is always reported.  Physical sensor isolation is opt-in because
    # a benchmark/replay run may intentionally contain large residuals that
    # should remain visible to the evaluator.
    enable_sensor_fault_isolation: bool = False
    integrity_failures_before_unsafe: int = 2
    integrity_recovery_before_reliable: int = 3

    # Integrity thresholds (for health monitoring)
    # The metric is the trace of the 6-DoF covariance (rad² + m²).  Values
    # below these defaults correspond to a normal initialized RGB-D track;
    # deployments should recalibrate them from a static validation sequence.
    delta_safe: float = 1e-2  # below this = Reliable
    delta_alarm: float = 5e-2  # above this = Unsafe

    # iSAM2 optimizer params
    relinearize_threshold: float = 0.01
    relinearize_skip: int = 1

    # Ablation switches (for experiments)
    use_adaptive_covariance: bool = True
    enable_online_calibration: bool = True
    enable_integrity_monitoring: bool = True

    # ---- legacy EKF parameters (used when backend="ekf") ----
    # (kept for backward compatibility)
    tau_obs: float = 0.5
    latency_init_samples: int = 30
    latency_alpha: float = 0.2
    tau_min: float = 0.02
    tau_max: float = 0.5
    tau_scale: float = 4.0
    w_feature: float = 0.4
    w_reproj: float = 0.4
    w_delay: float = 0.1
    w_occlusion: float = 0.1
    ref_features_hl2: float = 200.0
    ref_markers_optitrack: float = 8.0
    sigma_reproj: float = 2.0
    tau_delay: float = 0.2
    dt_fuse: float = 1.0 / 60.0


@dataclass
class BaselineConfig:
    # Topic names for online baseline outputs
    topic_best: str = "pose.baseline.best"
    topic_avg: str = "pose.baseline.avg"
    topic_single: str = "pose.baseline.single"

    # Single-camera baseline source key, e.g. "realsense/CAMERA_SERIAL"
    single_camera_source: str = ""

    # Use only DOPE detections (source ending with "/dope") for fair comparison
    dope_only: bool = True

    # Quality gating
    min_score: float = 0.05
    max_reproj_error_px: float = 20.0

    # Time window for multi-camera aggregation
    max_age_s: float = 0.20

    # Weighting for quality-aware averaging
    reproj_sigma_px: float = 4.0


@dataclass
class FrontendConfig:
    # Enable segmentation-based frontend observation pipeline.
    enable: bool = False

    # Detector backend: "yolo_seg" (default), extensible.
    detector: str = "yolo_seg"

    # Path to YOLO segmentation weights.
    model_path: str = "thirdparty/Ultralytics/models/yolov8n-seg.pt"
    device: str = "cpu"  # auto | cpu | cuda
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Optional class filter. Empty => keep all classes.
    target_classes: list[str] = field(default_factory=list)

    # Observation output topic for fusion input.
    topic_observations: str = "tracking/detections"

    # Depth processing
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0

    # Point-cloud filtering
    voxel_size_m: float = 0.01
    outlier_std_mul: float = 2.5
    outlier_min_neighbors: int = 8
    largest_cluster: bool = True

    # Optional bbox publish for debug.
    debug_topic: str = "frontend/seg_debug"


@dataclass
class FoundationPoseConfig:
    enable: bool = False
    foundationpose_root: str = "thirdparty/FoundationPose"
    hope_models_root: str = "assets/models/hope_original/cans"
    hope_scale_to_m: float = 0.01
    objects: list[str] = field(default_factory=list)
    source: str = ""
    initial_mask_path: str = ""
    register_iterations: int = 5
    track_iterations: int = 2
    score_tracking: bool = False  # Optional uncalibrated rank diagnostics; adds GPU work.
    # Optional RGB-D/CAD rendered-depth quality term.  It is disabled by
    # default because it adds a render pass; when enabled it is exported as
    # q_fit input to fusion, never as a calibrated probability by itself.
    depth_fit_quality_enable: bool = False
    depth_fit_quality_interval_s: float = 0.0
    warp_cache_path: str = "thirdparty/FoundationPose/.cache/warp"
    init_topic: str = "tracking/foundationpose/init"
    mask_topic: str = "tracking/segmentation_masks"
    class_to_object: dict[str, str] = field(default_factory=dict)
    output_topic: str = "tracking/detections"
    status_topic: str = "tracking/foundationpose/status"

    # Optional CAD-based mask recovery. CNOS runs on the same RGB frame and
    # periodically supplies a fresh mask to FoundationPose.register().
    recovery_enable: bool = False
    recovery_interval_s: float = 0.5
    recovery_conf_threshold: float = 0.45
    recovery_min_score_margin: float = 0.03
    recovery_min_mask_fraction: float = 3e-4
    recovery_confirmations: int = 2
    recovery_misses_before_lost: int = 2
    recovery_miss_confidence_decay: float = 0.35
    recovery_tracking_iou: float = 0.15
    recovery_candidate_iou: float = 0.10
    recovery_min_bbox_fill: float = 0.40
    recovery_cross_object_overlap: float = 0.50
    recovery_reservation_confidence: float = 0.25
    cnos_root: str = "thirdparty/CNOS"
    cnos_template_dir: str = "assets/cnos_templates/hope_cans"
    cnos_sam_model_type: str = "vit_b"
    cnos_sam_checkpoint: str = (
        "thirdparty/CNOS/pretrained/segment-anything/sam_vit_b_01ec64.pth"
    )
    cnos_dino_model_name: str = "dinov2_vitb14"
    cnos_dino_root: str = "thirdparty/dinov2"
    cnos_dino_checkpoint: str = (
        "thirdparty/dinov2/weights/dinov2_vitb14_pretrain.pth"
    )
    cnos_device: str = "cuda"
    cnos_image_width: int = 640
    cnos_max_proposals: int = 100


@dataclass
class EvaluationConfig:
    # Robot-derived GT file (JSONL/CSV adapted by experiments/evaluate.py)
    gt_path: str = ""

    # Failure thresholds
    fail_trans_m: float = 0.05
    fail_rot_deg: float = 10.0

    # Degradation label thresholds (for integrity precision/recall/FAR)
    degrade_trans_m: float = 0.03
    degrade_rot_deg: float = 5.0


@dataclass
class ExperimentConfig:
    # Experiment mode: normal | occlusion | async_delay | sensor_addition | degraded
    mode: str = "normal"

    # Per-camera timestamp delay injection, seconds
    delay_s_by_camera: dict[str, float] = field(default_factory=dict)

    # Occlusion simulation: randomly drop observations
    drop_prob_by_camera: dict[str, float] = field(default_factory=dict)

    # Degraded observation noise injected into pose (stddev)
    noise_trans_m: float = 0.0
    noise_rot_deg: float = 0.0

    # Online sensor addition scheduling
    camera_start_time_s: dict[str, float] = field(default_factory=dict)

    # New sensor uncertain extrinsic initialization noise
    new_sensor_init_trans_std_m: float = 0.0
    new_sensor_init_rot_std_deg: float = 0.0


@dataclass
class BackendConfig:
    # Backwards-compatible single-camera template (used for defaults / CLI overrides).
    camera: CameraConfig = field(default_factory=CameraConfig)

    # Configure camera serials for your own setup (or use --realsense-serials).
    realsense_cameras: list[CameraConfig] = field(default_factory=list)

    hl2: HL2Config = field(default_factory=HL2Config)
    ots: OTSConfig = field(default_factory=OTSConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    foundationpose: FoundationPoseConfig = field(default_factory=FoundationPoseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
