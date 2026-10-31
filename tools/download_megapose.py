"""Download the official pretrained MegaPose RGB-D model pair (not datasets)."""
import argparse
import hashlib
import json
import os
import ssl
from pathlib import Path
import urllib.request

import certifi


MODEL_IDS = ("coarse-rgb-906902141", "refiner-rgbd-288182519")
BASE_URL = "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models"


def main():
    if os.environ.get("CONDA_DEFAULT_ENV") != "mr_nav":
        raise RuntimeError("Activate conda mr_nav first")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/megapose_data")
    args = parser.parse_args()
    records = []
    for model in MODEL_IDS:
        for name in ("config.yaml", "checkpoint.pth.tar"):
            path = args.data_dir / "megapose-models" / model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{BASE_URL}/{model}/{name}"
            if not path.is_file():
                print(f"Downloading {url}", flush=True)
                partial = path.with_name(path.name + ".partial")
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(url, timeout=120, context=context) as response, partial.open("wb") as output:
                    expected = int(response.headers.get("Content-Length", 0))
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        count += len(chunk)
                    if expected and count != expected:
                        raise IOError(f"Incomplete download: {count}/{expected}")
                partial.replace(path)
            records.append({"url": url, "path": str(path), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"Ready: {path}", flush=True)
    (args.data_dir / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
