# Evaluation protocols

## Separate tracking from fusion

Each camera has its own tracker state. Object-to-camera poses are transformed
using that camera's extrinsics into a shared world frame before fusion.
Observation records retain object identity, camera identity, capture/arrival
times and available quality information. One fusion state is maintained per
object. The camera count is not hard-coded to the laboratory's three sensors.

H2O provides synchronized multi-view images, meshes and pose annotations.
Use actual tracker outputs as observations and annotations as evaluation targets.
Sending annotations themselves through fusion is only a consistency check.
GT-assisted first-frame initialization must be disclosed separately from scoring.

## Initialization and frontend controls

- FoundationPose: first-frame mask from annotations (`oracle_gt`) or independent
  masks (`mask_dir`); registration followed by per-frame tracking.
- M3T: region + depth modalities, initialized from the saved first FP pose per
  camera; subsequent frames do not consume FP/GT poses.
- MegaPose: the same initial-pose protocol, then RGB-D iterative refinement from
  its own previous estimate. This is a temporal-refinement adapter, not a claim
  to reproduce an unrelated published tracking benchmark.
- `oracle_gt_pose` is an alternative M3T/MegaPose protocol using the full initial
  GT pose. Do not pool it with the default initialization protocol.

The generic scores and depth-fit quality are not calibrated probabilities.
In particular, good depth agreement does not resolve all orientation ambiguity.
The H2O controls do not inject artificial noise, dropout or delay.

## Saved results

A frontend subject run contains `summary.json` and `takes/<group>_<take>/`.
Per-take files include `tracking_before_fusion.jsonl`, `ground_truth.jsonl`,
fused predictions, initialization information, metrics and mesh samples used
for scoring. Preserve the full run directory when copying a cache.

Cached evaluation uses identical observations and target times for all methods:
individual cameras, simple averages/quality selection, current fusion,
ablation variants, camera leave-one-out and independent robust/ESKF baselines.
Variant definitions live in `experiments/h2o/ablation.py`; independent baseline
settings are in `experiments/configs/fusion_baselines_v1.json`.

The H2O protocol enables checkpoint-based same-state revision. Recent-history
guards are optional; the laboratory protocol uses them with scene-specific
reference validity and timing checks. Enabling one setting is not evidence
that it improves every dataset. Production live defaults remain separate.

Report translation and rotation separately, means and tails, missing-output
rates and joint-threshold success/failure. Retain failed variants and their
missing suffixes; never select only frames on which a method succeeds.
Output coverage alone is not pose correctness. Prefer paired take-level
comparisons in addition to pooled frame metrics.

## Timing

`tools/benchmark_recorded_realtime.py` supports a bounded image replay profile.
Distinguish fusion-handler compute, frontend inference, output cadence and result
age. Replay/loopback results do not establish physical acquisition, network or
display latency. A full live deployment needs separate validation.
