from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for h5_path in sorted((args.session / "rgbd").glob("*.h5")):
        with h5py.File(h5_path, "r") as handle:
            colors = handle["color"]
            indices = sorted({0, len(colors) // 4, len(colors) // 2, 3 * len(colors) // 4, len(colors) - 1})
            frames = []
            for index in indices:
                image = np.asarray(colors[index])
                image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    image,
                    f"{h5_path.stem} frame {index}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                frames.append(image)
            sheet = np.vstack(frames)
            cv2.imwrite(str(args.output / f"{h5_path.stem}.jpg"), sheet)


if __name__ == "__main__":
    main()
