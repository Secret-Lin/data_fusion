"""Build project-owned M3T bindings using only the active mr_nav dependencies."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import pybind11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "mr_nav":
        raise RuntimeError("Activate conda mr_nav first")
    root = Path(__file__).resolve().parents[1]
    source = root / "adapters/m3t_native"
    build = source / "build"
    prefix = Path(sys.prefix).as_posix()
    # A case-duplicated PATH/Path inherited from PowerShell breaks MSBuild.
    environment = {k.upper(): v for k, v in os.environ.items()}
    cmake = str(Path(sys.prefix) / "Library/bin/cmake.exe")
    subprocess.run([cmake, "--fresh", "-S", str(source), "-B", str(build),
                    "-G", "Visual Studio 17 2022", "-A", "x64",
                    f"-DCMAKE_PREFIX_PATH={prefix}/Library;{prefix}",
                    f"-DPython_EXECUTABLE={sys.executable}",
                    f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"], env=environment, check=True)
    subprocess.run([cmake, "--build", str(build), "--config", "Release", "--target", "_m3t_native",
                    "--parallel", str(args.jobs)], env=environment, check=True)


if __name__ == "__main__":
    main()
