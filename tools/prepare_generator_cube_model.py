"""Explicitly convert a generator's right-handed Y-up cube for OpenCV PnP.

Unlike Unity handedness conversion, both Y and Z reverse here: this is a
proper rotation of the cube/tag frames, not a reflection of 3-D PnP points.
The input file is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml

from common.tag_dictionary import cube_dictionary


def convert_generator_model(data: dict) -> dict:
    cube_dictionary(str(data.get("dictionary", "")))
    convention = data.get("coordinate_convention", {})
    if "+Y Tag 0 top" not in str(convention.get("world_axes", "")) or "+Y top" not in str(convention.get("tag_axes", "")):
        raise ValueError("Expected generator cube/tag +Y top convention")
    if data.get("coordinate_system") not in (None, "y_up"):
        raise ValueError("Input is already normalized or has conflicting coordinate metadata")
    out = copy.deepcopy(data)
    basis = np.diag([1.0, -1.0, -1.0, 1.0])
    for entry in out["tags"].values():
        transform = np.asarray(entry["T_world_tag"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(transform[:3,:3]),1.0,atol=1e-5):
            raise ValueError("Invalid generator tag transform")
        entry["T_world_tag"] = (basis @ transform @ basis).tolist()
    out["coordinate_system"] = "y_down"
    out["coordinate_convention"] = {
        **convention,
        "world_axes": "+X Tag 0 right; +Y Tag 0 down; +Z inward through Tag 0 (right-handed)",
        "tag_axes": "+X right; +Y down; +Z inward from printed face (right-handed)",
        "T_normalized_cube_from_generator_cube": basis.tolist(),
        "T_normalized_tag_from_generator_tag": basis.tolist(),
        "conversion": "Proper Rx(pi) basis rotation; NOT the Unity Y reflection",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        raise FileExistsError("Output must be a new file, separate from the source")
    converted = convert_generator_model(yaml.safe_load(args.input.read_text(encoding="utf-8")))
    converted["source_model"] = str(args.input.resolve())
    converted["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(converted, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
