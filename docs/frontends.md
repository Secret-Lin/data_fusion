# Frontend setup

Cached fusion does not require these dependencies. The core `environment.yml`
does not claim to install a working CUDA stack. Install optional packages only
inside `mr_nav`; use upstream instructions appropriate to your OS, GPU and CUDA
toolchain. Windows inference was used during development; no clean Linux
frontend installation is claimed by this snapshot.

## FoundationPose

1. `python -m tools.setup_thirdparty --components FoundationPose`
2. Follow `thirdparty/FoundationPose/readme.md` for dependencies and official
   refine/score weights. This includes a compatible PyTorch/torchvision pair,
   CUDA, nvdiffrast, PyTorch3D, Warp, Open3D and the native `mycpp` extension.
   Builds and weights are not bundled. Keep all installs in `mr_nav`.
3. Retain the upstream weight/config directory layout inside FoundationPose.
4. Obtain H2O meshes or HOPE meshes; provide first-frame masks as described in
   the main README. Use `python -m tools.run_h2o_foundationpose --help`.

CNOS recovery is optional for laboratory replay, not the default H2O protocol.
Install `CNOS` and `dinov2` with the setup helper and install the SAM dependency.
Supply the SAM/DINOv2 checkpoints and rendered CAD templates configured in
`config.py`; `tools/render_cnos_hope_templates.py` supports template generation.
These assets are not automatically downloaded by the source helper.

## M3T

1. `python -m tools.setup_thirdparty --components 3DObjectTracking`
2. On Windows, install Visual Studio 2022 C++ build tools and CMake. The native
   bridge needs pybind11, Eigen3, OpenCV (including xfeatures2d), GLEW, GLFW and
   OpenGL. Native libraries must match the active conda runtime.
3. In `mr_nav`, install the native build dependencies if not already available:
   `conda install -c conda-forge cmake pybind11 eigen glew glfw opencv`.
4. Run `python -m tools.build_m3t`. The current build helper selects the Visual
   Studio generator; other platforms need an appropriate CMake invocation.

Only M3T's region + depth path is enabled. This is not a separate ICG benchmark.
Model caches are generated locally and ignored by Git.

## MegaPose

1. `python -m tools.setup_thirdparty --components megapose6d`
2. Follow the exported upstream setup instructions and review
   `adapters/megapose_requirements.txt` for the tested extra inference packages.
   That file is incremental, not a standalone dependency lock. The frontend
   also requires torch/torchvision, OmegaConf, Open3D, PyArrow and related
   upstream dependencies, in addition to the core environment.
3. `python -m tools.download_megapose` obtains the official RGB-D model pair
   into the ignored model-data directory and records download provenance.
4. Use `python -m tools.run_h2o_megapose --help` and supply the FP seed run.

The local NumPy/SciPy transform fallback is used if Pinocchio is unavailable.
Renderer changes and their limitations are described in `thirdparty/README.md`.

## Other interfaces

DOPE requires its own source and object-specific weights; pass `--dope-root` and
`--dope-weights` explicitly. Segmentation requires Ultralytics and the selected
weights. Sensor SDKs are optional and documented in `docs/inputs.md`.

## Validation levels

Core unit tests and cached-fusion smoke checks can run without installing any
tracker. Additional tests may import PyTorch or read optional vendor source;
install the relevant component before running those tests. The release was
checked in the existing `mr_nav` environment, not by rebuilding every GPU
frontend from scratch. No weights or compiled environment are redistributed.
