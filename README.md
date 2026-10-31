# Quality-Adaptive Multi-Camera 6D Object Pose Fusion with Revisable Cross-View Consensus

[Code repository](https://github.com/Secret-Lin/data_fusion) · Source snapshot refreshed 2026-10-31.

Research implementation of pose-level multi-camera fusion. The backend combines
object-pose observations from fixed or moving cameras using quality-adaptive
uncertainty, revisable cross-view support and temporal constraints.

## What this repository provides

- **Quality-adaptive robust temporal fusion:** per-object incremental GTSAM
  estimation, robust pose factors, cross-view consistency, same-state revision
  and optional bounded recent-history protection.
- **Multiple tracking frontends:** FoundationPose, M3T and MegaPose adapters and
  H2O evaluation runners; the backend also retains DOPE and RGB-D segmentation
  interfaces. Frontends produce observations; they are not the proposed fusion
  algorithm, and their confidence values are not automatically interchangeable.
- **Asynchronous incremental updates:** capture-time association is distinct
  from arrival order. An update may be revised when another view arrives.
  Fusion compute time, frontend inference time and end-to-end latency are
  different quantities; a repeated output at 30 Hz is not 30 new estimates.

Experiments compare individual cameras, simple fusion, independent pose-level
baselines and controlled ablations on the same saved observations. Benefits are
assessed over sequences, including error tails and failures; improvement is not
guaranteed on every frame, sequence or frontend. Consistent bias, ambiguous
orientations and loss of all useful observations remain limitations.

## Reproduction scope

This release provides the backend, frontend adapters, data readers and evaluation
entry points. It does not bundle the original datasets, object meshes, pretrained
weights, saved frontend observations or the manuscript's local evidence archive.
Obtain the data and model assets separately, following their original terms, then
use the commands below to generate observations and compare fusion methods.

Reuse the same saved observations and target groups across fusion comparisons;
see [evaluation protocols](docs/evaluation.md). Unit tests and the small format
fixture check implementation behavior. They do not regenerate the paper's full
experimental results or establish end-to-end real-time performance.

## Installation

Run commands from the repository root in the **`mr_nav`** conda environment.
The core recipe does not install GPU frontends or hardware SDKs.

```powershell
conda env create -f environment.yml
conda activate mr_nav
```

If `mr_nav` already exists, activate it and review dependencies rather than
recreating or blindly upgrading a working CUDA environment. Development used
Windows, Python 3.10, conda-forge GTSAM 4.2 and NumPy 2.2. The environment file is
a core dependency recipe, not a complete platform-independent lock file.
See [frontend setup](docs/frontends.md) for inference dependencies and weights.

## Quick check: fusion without cameras or GPU inference

```powershell
python -m pytest -q tests/test_pose_consensus.py tests/test_fusion_world_revision.py tests/test_fusion_supported_history.py tests/test_fusion_singleton_guard.py tests/test_independent_fusion_baselines.py
```

An optional format/coordinate smoke check creates a small deterministic fixture:

```powershell
python -m experiments.h2o.smoke_fixture --output artifacts/smoke/input --frames 12
python -m experiments.h2o.benchmark --sequence artifacts/smoke/input/subject_smoke/h1/0 --output artifacts/smoke/results_0 --max-frames 12
```

This fixture checks plumbing only. It is not H2O experimental evidence and is not
an artificial-corruption accuracy experiment. Use a new output path on reruns.

## H2O evaluation

Obtain RGB-D sequences, object meshes and annotations from the
[official H2O dataset](https://github.com/taeinkwon/h2odataset) under its terms.
This repository does not redistribute them. Suggested local layout:

```text
data/h2o/
  object.zip
  subject1/<group>/<take>/cam0/...cam4/
  subject2/...
  subject3/...
  subject4/...
```

### 1. Generate frontend observations

After [installing FoundationPose](docs/frontends.md), run a bounded check:

```powershell
python -m tools.run_h2o_foundationpose --subject-root data/h2o/subject1 --mesh-source data/h2o/object.zip --take h1/0 --cameras all --max-frames 30 --depth-fit-quality --init-mode oracle_gt --output-parent results/foundationpose
```

`oracle_gt` uses an annotation-derived **first-frame mask**. This is an explicitly
GT-assisted initialization protocol, not independent object detection. For
independent masks, use `--init-mode mask_dir --mask-root <directory>` containing
`<camera>/<frame:06d>.png`. No later GT is used to guide fusion.

Omit `--take` to process all valid takes in a subject; `--max-frames 0` removes
the frame limit. All valid cameras are used by default. Windows all-subject
launchers are available for each frontend:

```powershell
.\tools\run_h2o_all_subjects.ps1 -DataRoot "data/h2o"
.\tools\run_h2o_m3t_all_subjects.ps1 -DataRoot "data/h2o" -SeedRoot "data/h2o/results_all_subjects"
.\tools\run_h2o_megapose_all_subjects.ps1 -DataRoot "data/h2o" -SeedRoot "data/h2o/results_all_subjects"
```

Run FoundationPose first. M3T and MegaPose use each camera's saved first FP pose
under the shared-initial-estimate protocol, then track independently. The default
H2O runners do not redetect lost targets. See [protocol and outputs](docs/evaluation.md).

### 2. Re-fuse and ablate cached observations

No GPU, RGB-D reloading or frontend installation is needed for cached fusion.
Replace the example run directory with the actual `results_N` directory:

```powershell
python -m tools.run_h2o_foundationpose_ablation --run-dir results/foundationpose/results_0 --output-parent results/ablation --leave-one-out
python -m tools.run_h2o_fusion_baselines --run-dir results/foundationpose/results_0 --frontend foundationpose --output-parent results/baselines --workers 4
```

The historical `foundationpose_ablation` module name also accepts saved M3T and
MegaPose observations without rerunning or changing their frontend. Leave-one-out
removes each camera in turn; it does not remove a subject or object class.
Outputs use incrementing `results_N` directories.

## Laboratory RGB-D replay and live interfaces

`backend_main.py` retains replay and live adapters; `standalone/` contains the
Quest/RealSense recorder and tag-cube tools. Start with
`python backend_main.py --help` and [input requirements](docs/inputs.md).
Data-dependent calibration, masks, camera identifiers and reference alignment
must be supplied for a new scene. No laboratory data or measured extrinsics are
included. The Unity navigation/bridge placeholders are not complete applications.

## Repository layout

| Path | Purpose |
|---|---|
| `core/`, `managers/` | Typed events, cross-view consensus and fusion |
| `adapters/` | Tracking and sensor adapters, M3T native bridge |
| `experiments/h2o/` | Dataset reader, scoring and cached fusion |
| `experiments/fusion_baselines.py` | Independent pose-level baselines |
| `tools/` | Frontend runners, ablations, calibration and analysis |
| `standalone/` | RGB-D recording and tag-cube utilities |
| `tests/` | Unit and integration tests |
| `thirdparty/` | Pinned sources, compatibility patches and notices |

## Scope, attribution and license

The repository excludes datasets, model weights, local experiment outputs,
calibration results, manuscript drafts, private machine settings and Git history.
Existing fusion algorithms and experiment defaults are preserved in this public
snapshot; path defaults and optional-dependency packaging are made portable.

See [third-party notices](thirdparty/README.md). Upstream code, weights and data
retain their own licenses, including research-use restrictions where applicable.
Project-authored code is released under the [MIT License](LICENSE). This license
does not relicense third-party sources, compatibility-patch context, weights or
datasets. Consult their original terms before use or redistribution.
