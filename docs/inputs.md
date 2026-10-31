# Laboratory inputs

Run `python backend_main.py --help` for current arguments. Optional SDKs are
required only for the relevant sensor/frontend. Do not start the default live
configuration before setting your camera identifiers and network addresses.

## Recorded RGB-D

```text
session/
  camera_metadata.json
  rgbd/<camera>.h5
  quest/<camera>.h5
  tag_cube_extrinsics.json   # fixed-camera extrinsics, when supplied
  tag_cube_pose.json         # Quest world alignment, when supplied
```

The recorder in `standalone/record_realsense_and_quest_rgbd_dpt2.py` describes
its HDF5 schema. RGB-D replay reads color, depth, timestamps, intrinsics and
camera-pose information where available. Moving-camera world poses are not
automatically tag-cube world poses: an explicit alignment is required.

The project uses Y-down coordinates at its input boundary. Unity Y-up data must
be converted once, with the conversion explicitly recorded. Fixed and moving
sensors require extrinsics. If the replay extrinsic file is absent, the backend
can estimate it by scanning tag-cube observations; an unresolvable calibration
is an error, not an identity transform. Fixed cameras use their extrinsics;
visible tags on moving cameras may correct drift.

Supply the matching tag family, physical dimensions and face layout. The two
YAML files in `standalone/` illustrate laboratory cube models; they are not a
universal calibration. `cube_model_generator.py --help` describes dictionary
selection. Do not reuse another setup's measured extrinsics or timing offsets.

## Models and initialization

HOPE assets are obtained separately. The expected layout is
`assets/models/hope_original/cans/<ObjectName>/google_16k/textured_simple.obj`,
with its material and texture files. Check units; HOPE loading defaults to
centimetres-to-metres scaling, while H2O meshes are already in metres.
Provide independent initial masks or configure CNOS recovery and its CAD
templates. Neither masks nor calibrated camera/reference poses are bundled.

## Optical reference

OTS is an evaluation reference, not an input to the proposed object fusion.
Laboratory reference evaluation requires both OTS-to-world and rigid-body-to-
object registration, valid rigid-body tracking, and session-specific clock
alignment. `tools/real_ots_reference.py` and the real-scene runners retain the
reference validity checks. Session-named batch scripts reproduce the laboratory
protocol only after the corresponding data and calibration artifacts are
provided; they are not zero-configuration examples for a different scene.

## Hardware interfaces

RealSense requires pyrealsense2 and the device runtime. Quest requires the
matching streaming application/protocol and camera-pose metadata. HoloLens
requires HL2SS. NatNet/OTS and Unity bridge components have implementation
limitations; this repository does not bundle the Unity application or provide
a complete surgical-navigation deployment.

## Shipped cube-model examples

The two YAML examples retain their explicit `y_up` model-frame declaration,
ideal tag geometry and dictionary. They do not contain camera extrinsics.
The source camera serial has been removed; geometric fit metadata is retained.
Respect the declared frame when preparing a Y-down experiment;
`python -m tools.prepare_generator_cube_model --help` describes the
separate model-conversion utility. Do not only relabel the coordinate field.
