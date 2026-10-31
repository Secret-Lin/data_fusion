"""Export pinned optional frontend sources without copying Git history.

This downloads source only: no pip/conda installs, model weights or builds.
Existing nonempty directories are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile


def export_source(spec: dict, thirdparty: Path, name: str) -> Path:
    relative = PurePosixPath(spec.get('destination', name))
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
        raise ValueError('Unsafe dependency destination')
    destination = (thirdparty / str(relative)).resolve()
    if not destination.is_relative_to(thirdparty.resolve()):
        raise ValueError('Dependency destination escapes thirdparty')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'Refusing to overwrite {destination}')
    patch = thirdparty / spec['patch'] if spec.get('patch') else None
    if patch and hashlib.sha256(patch.read_bytes()).hexdigest() != spec['patch_sha256']:
        raise ValueError(f'Patch checksum mismatch: {patch}')
    with tempfile.TemporaryDirectory(prefix='pose-fusion-source-') as scratch:
        checkout = Path(scratch) / 'checkout'
        subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', spec['revision']], check=True)
        actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec['revision']:
            raise ValueError(f'Revision mismatch for {name}')
        archive = Path(scratch) / 'source.zip'
        subprocess.run(['git', '-C', str(checkout), 'archive', '--format=zip',
                        f'--output={archive}', 'HEAD'], check=True)
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '.git' in path.parts or ':' in item.filename:
                    raise ValueError(f'Unsafe archive path: {item.filename}')
                if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f'Symlink needs manual installation: {item.filename}')
            destination.mkdir(parents=True, exist_ok=True)
            source.extractall(destination)
        if patch:
            environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(thirdparty.resolve()))
            subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=destination, env=environment, check=True)
            subprocess.run(['git', 'apply', str(patch.resolve())], cwd=destination, env=environment, check=True)
        (destination / '.source-provenance.json').write_text(
            json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    return destination


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    thirdparty = root / 'thirdparty'
    sources = json.loads((thirdparty / 'sources.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components', nargs='+', choices=list(sources))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(sources, indent=2))
        return
    if not args.components:
        parser.error('Select --components, or use --list; no implicit downloads')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'mr_nav':
        raise RuntimeError('Activate conda mr_nav')
    selected = list(dict.fromkeys(args.components))
    if 'megapose6d' in selected:
        selected = [n for n in selected if n != 'bop_toolkit_challenge'] + ['bop_toolkit_challenge']
    for name in selected:
        print(f'{name}: {export_source(sources[name], thirdparty, name)}', flush=True)


if __name__ == '__main__':
    main()
