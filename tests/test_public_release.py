"""Packaging checks without a dataset, GPU frontend or remote download."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.setup_thirdparty import export_source
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def test_cached_fusion_fingerprints_allow_missing_optional_vendor():
    hashes = code_fingerprints()
    assert len(hashes['managers/fusion_manager.py']) == 64
    assert hashes['thirdparty/FoundationPose/Utils.py'] == 'not_installed' or len(hashes['thirdparty/FoundationPose/Utils.py']) == 64


def test_source_export_refuses_existing_and_escaping_destinations(tmp_path):
    existing = tmp_path / 'vendor'
    existing.mkdir()
    (existing / 'user.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        export_source({}, tmp_path, 'vendor')
    with pytest.raises(ValueError):
        export_source({'destination': '../escape'}, tmp_path, 'vendor')
    assert (existing / 'user.txt').read_text() == 'preserve'


def test_source_export_checks_patch_before_network(tmp_path):
    (tmp_path / 'change.patch').write_text('not a valid patch')
    with pytest.raises(ValueError, match='checksum'):
        export_source({'patch': 'change.patch', 'patch_sha256': '0' * 64}, tmp_path, 'vendor')


@pytest.mark.skipif(shutil.which('git') is None, reason='Git not installed')
def test_source_export_keeps_patch_but_not_git_history(tmp_path):
    upstream = tmp_path / 'local upstream'
    upstream.mkdir()
    subprocess.run(['git', 'init', str(upstream)], check=True, capture_output=True)
    (upstream / 'sample.txt').write_text('before\n')
    subprocess.run(['git', '-C', str(upstream), 'add', 'sample.txt'], check=True)
    subprocess.run(['git', '-C', str(upstream), '-c', 'user.name=Release Test',
                    '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    thirdparty = tmp_path / 'public copy' / 'thirdparty'
    thirdparty.mkdir(parents=True)
    patch = b'diff --git a/sample.txt b/sample.txt\n--- a/sample.txt\n+++ b/sample.txt\n@@ -1 +1 @@\n-before\n+after\n'
    (thirdparty / 'change.patch').write_bytes(patch)
    destination = export_source({'url': str(upstream), 'revision': revision,
                                 'patch': 'change.patch', 'patch_sha256': hashlib.sha256(patch).hexdigest()},
                                thirdparty, 'vendor')
    assert (destination / 'sample.txt').read_text() == 'after\n'
    assert not list(destination.rglob('.git'))
    assert (upstream / 'sample.txt').read_text() == 'before\n'
