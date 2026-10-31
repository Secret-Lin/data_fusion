# Public source release

Refreshed on 2026-10-31 from the working project. The initial public-source
snapshot was prepared on 2026-09-29. This remains a source-only research release,
not a packaged GPU runtime or a redistribution of experimental datasets.

Repository: <https://github.com/Secret-Lin/data_fusion>.

## Research scope

- Quality-adaptive robust temporal fusion is the research backend. Tracking
  frontends provide pose observations and available quality information.
- FoundationPose, M3T and MegaPose adapters and H2O evaluation entry points are
  retained, alongside the existing DOPE and RGB-D segmentation interfaces.
- Capture-time grouping, sequential arrivals and eligible same-state revisions
  remain distinct. Incremental fusion cost is not end-to-end system latency.
- Evaluation compares fusion methods on the same saved observations and target
  groups. Translation, rotation, tails, failures and sequence-level variation
  remain separate outcomes; gains are not guaranteed for every frontend or take.

## Public packaging

- Local data layouts use `data/h2o` and `data/laboratory`; users supply their own
  sensor identifiers, addresses, masks and measured calibration.
- The core environment recipe is `environment.yml`, named `mr_nav`. Optional
  GPU frontends, sensor SDKs, checkpoints and native builds are installed
  separately under their upstream instructions and licenses.
- Third-party adaptations are exported as checksummed patches against pinned
  upstream revisions, with license and provenance records. They are not
  relicensed under the project's MIT license.
- Raw H2O/HOPE data, laboratory RGB-D or optical-reference data, measured
  extrinsics, model weights, local run outputs and manuscript working files
  are excluded. The repository contains reproduction entry points rather than
  the paper's complete frozen experiment archive.
- The original project's Git history is not imported. The public repository
  starts from the independently prepared source release.

## Initialization and evaluation

H2O FoundationPose uses either an annotation-derived first-frame mask
(`oracle_gt`) or independently supplied masks (`mask_dir`). The default M3T and
MegaPose comparisons use the corresponding first saved FoundationPose poses,
then track independently. These initialization choices must be disclosed; no
later reference pose is used to select fusion support or update the state.
See [evaluation protocols](evaluation.md) and [frontend setup](frontends.md).

## Validation records

The initial 2026-09-29 packaging check recorded **211 tests passed and 10
skipped** in the existing `mr_nav` environment. Those counts describe the
initial snapshot, not a new result for this refresh. The skips covered optional
FoundationPose source, HOPE asset and laboratory-report dependencies that were
not redistributed.

Core regression checks and bounded format/coordinate checks validate code and
packaging behavior. They do not replace a clean GPU-stack installation, a
full-data inference run or live-device latency validation. This release does
not claim new experimental performance merely because the source was refreshed.

## 2026-10-31 synchronization audit

The current working `core/`, `managers/`, `adapters/` and `common/` files
were compared with this public snapshot and are byte-identical. Existing
portable path defaults and optional-dependency handling are preserved;
this refresh does not introduce a new fusion algorithm or change thresholds.
The two example cube-model YAMLs omit only the calibration camera serial;
their geometry and declared model frames are unchanged. The example scope
and explicit frame preparation are documented in [inputs](inputs.md).
The root license is the previously selected MIT license for project-authored
code; upstream terms are unchanged.

## Refresh validation — 2026-10-31

In the existing `mr_nav` environment, the refreshed public copy passes
**211 tests, with 10 explicit skips**: eight require optional FoundationPose
source, one requires HOPE assets and one requires laboratory report artifacts.
All eight CLI/source-list checks and all 20 PowerShell syntax checks pass.
A 12-frame, five-camera deterministic format fixture also completes. It is
a plumbing check, not an accuracy experiment or a new benchmark result.
FoundationPose and MegaPose patches match the pinned working revisions
and pass reverse-application checks. Machine-readable details are in
[release_validation.json](release_validation.json).
