from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tools.download_h2o import (
    _extract_archive,
    _files_for_mode,
    _total_size_from_headers,
)


def test_h2o_downloader_can_select_one_full_subject() -> None:
    assert _files_for_mode("all", [1]) == [
        "object.zip",
        "label_split.zip",
        "subject1_v1_1.tar.gz",
    ]


def test_h2o_pose_and_full_archives_are_distinct() -> None:
    assert _files_for_mode("pose", [2])[-1] == "subject2_pose_v1_1.tar.gz"
    assert _files_for_mode("all", [2])[-1] == "subject2_v1_1.tar.gz"


def test_content_range_reports_full_resumed_size() -> None:
    assert (
        _total_size_from_headers(
            offset=100,
            status_code=206,
            content_length=900,
            content_range="bytes 100-999/1000",
        )
        == 1000
    )


def test_content_range_rejects_wrong_resume_offset() -> None:
    with pytest.raises(IOError, match="offset mismatch"):
        _total_size_from_headers(
            offset=100,
            status_code=206,
            content_length=950,
            content_range="bytes 50-999/1000",
        )


def test_selective_take_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "subject1_v1_1.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for take, payload in (("0", b"wanted"), ("1", b"other")):
            name = f"subject1/h1/{take}/cam0/cam_pose/000000.txt"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))

    output = tmp_path / "dataset"
    _extract_archive(
        archive,
        output,
        member_prefix="subject1/h1/0",
    )

    extracted = output / "subject1/h1/0/cam0/cam_pose/000000.txt"
    assert extracted.read_bytes() == b"wanted"
    assert not (output / "subject1/h1/1").exists()
