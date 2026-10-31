# Optional third-party sources

This directory contains source provenance, local compatibility patches and
upstream license notices, not full checkouts, Git history, weights or binaries.
Exact revisions are recorded in [sources.json](sources.json).

| Component | Upstream | Use |
|---|---|---|
| FoundationPose | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | RGB-D registration/tracking |
| M3T | [DLR-RM/3DObjectTracking](https://github.com/DLR-RM/3DObjectTracking) | Region + depth tracking |
| MegaPose | [megapose6d/megapose6d](https://github.com/megapose6d/megapose6d) | RGB-D pose refinement |
| BOP toolkit | [ylabbe/bop_toolkit_challenge](https://github.com/ylabbe/bop_toolkit_challenge) | MegaPose dependency |
| CNOS | [nv-nguyen/cnos](https://github.com/nv-nguyen/cnos) | Optional CAD-based mask recovery |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | CNOS descriptors |
| HL2SS | [jdibenes/hl2ss](https://github.com/jdibenes/hl2ss) | Optional HoloLens interface |

From the project root, activate `mr_nav`, then install only the components needed:

```powershell
python -m tools.setup_thirdparty --list
python -m tools.setup_thirdparty --components FoundationPose
python -m tools.setup_thirdparty --components 3DObjectTracking
python -m tools.setup_thirdparty --components megapose6d
```

The helper requires Git/network access, checks out the pinned revision in a
temporary directory and exports it without `.git`. MegaPose also exports its
pinned BOP dependency. It verifies and applies the corresponding patch. It never
overwrites a nonempty dependency directory, installs packages or fetches weights.
Interrupted exports must be inspected before retrying. Follow each upstream's
build and weight instructions after obtaining the source. A missing dependency
is allowed for cached fusion and is explicitly recorded as `not_installed` in
its provenance; it is not a verified frontend hash.

## Local adaptations

- **FoundationPose:** platform-neutral temporary mesh path, MSVC build guards,
  OpenCV import ordering and blockwise mesh diameter calculation. The latter
  uses the same sampled points and maximum pairwise distance with bounded
  temporary memory; it does not change trained weights.
- **MegaPose:** Windows path/offscreen rendering compatibility, NumPy 2 types,
  optional Pinocchio replacement for inference transforms and bounded worker
  waits/shutdown. Windows offscreen rendering disables MSAA; this is a rendering
  difference from upstream. The fallback is not a Pinocchio dynamics library.
- **CNOS, DINOv2, M3T:** source is exported unmodified; wrappers are in `adapters/`.
- **HL2SS:** original client source is exported. Local, unrelated viewer/demo
  customizations from the development checkout are not included.

The MIT license at the project root applies only to project-authored code.
Upstream sources and patch context retain their own terms. In particular,
FoundationPose and MegaPose are not relicensed as MIT by this repository.
Preserved license texts are in `licenses/`; fetched checkouts also include their
original notices. Data and model weights may have separate restrictions.
DOPE, SAM, Ultralytics, GTSAM, PyTorch and other installed dependencies retain
their respective licenses and must be obtained separately.
