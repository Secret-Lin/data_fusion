from pathlib import Path

from managers.record_manager import RecordManager


def test_next_result_name_increments_highest_existing_index(tmp_path: Path) -> None:
    (tmp_path / "results_0").mkdir()
    (tmp_path / "results_2").mkdir()
    (tmp_path / "results_old_name").mkdir()

    assert RecordManager._next_result_name(tmp_path) == "results_3"
