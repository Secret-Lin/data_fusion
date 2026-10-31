from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


BASE_URL = "https://h2odataset.ethz.ch/data/dataset"
MODES = ("all", "ego", "pose")


def _files_for_mode(mode: str, subjects: list[int]) -> list[str]:
    suffix = {"all": "v1_1", "ego": "ego_v1_1", "pose": "pose_v1_1"}[mode]
    return [
        "object.zip",
        "label_split.zip",
        *(f"subject{subject}_{suffix}.tar.gz" for subject in subjects),
    ]


def _download_file(
    *,
    destination: Path,
    filename: str,
    username: str,
    password: str,
    max_retries: int,
    retry_wait_s: float,
) -> None:
    url = f"{BASE_URL}/{filename}"
    target = destination / filename
    partial = target.with_suffix(target.suffix + ".part")

    for attempt in range(max_retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {
            "Accept-Encoding": "identity",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        }
        try:
            with requests.get(
                url,
                auth=(username, password),
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code == 401:
                    raise RuntimeError("H2O rejected the username/password")
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored the Range request for {filename}; "
                        "the existing partial file was preserved"
                    )

                content_length = int(response.headers.get("content-length", 0))
                content_range = str(response.headers.get("content-range", ""))
                total = _total_size_from_headers(
                    offset=offset,
                    status_code=int(response.status_code),
                    content_length=content_length,
                    content_range=content_range,
                )
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream, tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        progress.update(len(chunk))

            downloaded = partial.stat().st_size
            if total is not None and downloaded != total:
                raise IOError(
                    f"Incomplete {filename}: downloaded {downloaded} of {total} bytes"
                )
            partial.replace(target)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries + 1} attempts: {filename}"
                ) from exc
            wait_s = min(float(retry_wait_s) * (2 ** min(attempt, 4)), 60.0)
            saved = partial.stat().st_size if partial.exists() else 0
            print(
                f"[retry {attempt + 1}/{max_retries}] {filename}: {exc}; "
                f"saved={saved} bytes, waiting {wait_s:.1f}s"
            )
            time.sleep(wait_s)


def _total_size_from_headers(
    *, offset: int, status_code: int, content_length: int, content_range: str
) -> int | None:
    if status_code == 206:
        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
        if match is None:
            raise IOError(f"Invalid Content-Range header: {content_range!r}")
        start, end, total_text = match.groups()
        if int(start) != int(offset):
            raise IOError(
                f"Resume offset mismatch: requested {offset}, server returned {start}"
            )
        if int(end) < int(start):
            raise IOError(f"Invalid Content-Range header: {content_range!r}")
        return None if total_text == "*" else int(total_text)
    return int(content_length) if content_length > 0 else None


def _safe_archive_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Unsafe archive member: {member_name}") from exc
    return target


def _extract_archive(
    archive: Path,
    destination: Path,
    *,
    member_prefix: str | None = None,
) -> None:
    normalized_prefix = (
        str(member_prefix).replace("\\", "/").strip("/")
        if member_prefix
        else None
    )
    marker_suffix = (
        "." + normalized_prefix.replace("/", "_") if normalized_prefix else ""
    )
    marker = destination / ".h2o_extracted" / (
        f"{archive.name}{marker_suffix}.done"
    )
    if marker.exists():
        print(f"[skip extracted] {archive.name}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = _safe_archive_target(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    elif archive.name.lower().endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            found_selected_take = False
            for member in bundle:
                member_name = member.name.replace("\\", "/").strip("/")
                selected = normalized_prefix is None or (
                    member_name == normalized_prefix
                    or member_name.startswith(normalized_prefix + "/")
                )
                if not selected:
                    # Official H2O archives group every take contiguously. Once
                    # the selected take has ended, stop decompressing the rest
                    # of the 180+ GB subject archive.
                    if normalized_prefix is not None and found_selected_take:
                        break
                    continue
                found_selected_take = True
                target = _safe_archive_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(
                        f"Refusing archive link member: {member.name}"
                    )
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            if normalized_prefix is not None and not found_selected_take:
                raise FileNotFoundError(
                    f"Take {normalized_prefix!r} not found in {archive}"
                )
    else:
        raise ValueError(f"Unsupported H2O archive: {archive}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-capable downloader for the official H2O archives"
    )
    parser.add_argument("--dest", default=r"data/h2o")
    parser.add_argument("--mode", choices=MODES, default="pose")
    parser.add_argument(
        "--subjects",
        default="1,2,3,4",
        help="Comma-separated H2O subject numbers; use 1 for the first full-data test",
    )
    parser.add_argument("--username", default=os.environ.get("H2O_USERNAME", ""))
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract archives into DEST/dataset after download (default: disabled)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Do not authenticate or download; extract already downloaded archives",
    )
    parser.add_argument(
        "--extract-dir",
        default="",
        help="Extraction directory (default: DEST/dataset)",
    )
    parser.add_argument(
        "--take",
        default="",
        help="Extract only one take, e.g. subject1/h1/0 (tar archives only)",
    )
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--retry-wait-s", type=float, default=5.0)
    args = parser.parse_args()

    try:
        subjects = sorted(
            {
                int(value.strip())
                for value in str(args.subjects).split(",")
                if value.strip()
            }
        )
    except ValueError as exc:
        raise ValueError("--subjects must contain comma-separated integers 1..4") from exc
    if not subjects or any(subject not in {1, 2, 3, 4} for subject in subjects):
        raise ValueError("--subjects must select at least one value from 1,2,3,4")
    filenames = _files_for_mode(args.mode, subjects)
    take = str(args.take).replace("\\", "/").strip("/")
    take_parts: list[str] = []
    if take:
        take_parts = take.split("/")
        if (
            len(take_parts) != 3
            or take_parts[0] not in {f"subject{i}" for i in range(1, 5)}
            or not take_parts[2].isdigit()
        ):
            raise ValueError("--take must look like subject1/h1/0")
        take_subject = int(take_parts[0].replace("subject", ""))
        if take_subject not in subjects:
            raise ValueError(
                f"--take selects subject{take_subject}, but --subjects excludes it"
            )

    destination = Path(args.dest).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not args.extract_only:
        username = str(args.username).strip() or input("H2O username: ").strip()
        password = os.environ.get("H2O_PASSWORD") or getpass.getpass("H2O password: ")
        if not username or not password:
            raise RuntimeError("H2O username and password are required")
        for filename in filenames:
            target = destination / filename
            if target.exists():
                print(f"[skip] {target}")
                continue
            _download_file(
                destination=destination,
                filename=filename,
                username=username,
                password=password,
                max_retries=max(int(args.max_retries), 0),
                retry_wait_s=max(float(args.retry_wait_s), 0.1),
            )
    if args.extract:
        extract_dir = (
            Path(args.extract_dir).expanduser().resolve()
            if str(args.extract_dir).strip()
            else destination / "dataset"
        )
        for filename in filenames:
            archive = destination / filename
            if not archive.is_file():
                raise FileNotFoundError(f"H2O archive not found: {archive}")
            prefix = take if take and archive.name.endswith(".tar.gz") else None
            if prefix and not archive.name.startswith(take_parts[0] + "_"):
                continue
            _extract_archive(archive, extract_dir, member_prefix=prefix)


if __name__ == "__main__":
    main()
