import json
from dataclasses import dataclass

from common.run_manifest import build_run_manifest, write_manifest


@dataclass
class DemoConfig:
    threshold: float = 0.25


def test_manifest_fingerprints_artifact_and_writes_atomically(tmp_path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model-v1")
    manifest = build_run_manifest(
        cfg=DemoConfig(),
        workspace=tmp_path,
        artifact_paths=[artifact],
        runtime={"input_mode": "replay"},
    )
    output = tmp_path / "run_manifest.json"
    write_manifest(output, manifest)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["configuration"]["threshold"] == 0.25
    assert loaded["runtime"]["input_mode"] == "replay"
    assert loaded["artifacts"][0]["sha256"]
    assert not output.with_suffix(".json.tmp").exists()
