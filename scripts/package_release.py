"""Build a scoped source/data ZIP without virtualenv, caches or unrelated files."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from flysoc import __version__
from flysoc.artifacts import run_id, sha256, write_json
from flysoc.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-data', action='store_true', help='Include current local research data and trusted model')
    args = parser.parse_args()
    files: set[Path] = set()
    for name in ['README.md', 'LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'CITATION.cff', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', 'requirements.txt',
                 'requirements-lock.txt', '.gitignore', 'Start-BrainLab.ps1', 'Start-BrainLab.cmd']:
        files.add(ROOT / name)
    for folder in ['src/flysoc', 'scripts', 'tests', 'config', 'docs', '.github']:
        files.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    if args.with_data:
        run = json.loads((ROOT / 'results/latest.json').read_text())['run_id']
        validation = json.loads((ROOT / 'results/latest_brain_validation.json').read_text())['path']
        for folder in ['data/synthetic', 'data/connectome/fafb783', f'models/{run}',
                       f'results/{run}', f'results/{validation}', 'results/validation']:
            files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix != '.part')
        files.update([ROOT / 'results/latest.json', ROOT / 'results/latest_brain_validation.json'])
    for path in files:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Archive input outside the FlySOC workspace')
    records = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(files)}
    destination = ROOT / 'results/releases' / f'FlySOC-v{__version__}-{run_id("lab")}.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'version': __version__, 'includes_data': args.with_data, 'files': records,
                'installation': 'Create .venv using scripts/bootstrap.ps1, then launch Start-BrainLab.cmd. '
                                'Python and pip dependencies are not included.',
                'third_party_notices': 'THIRD_PARTY_NOTICES.md'}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(files):
            archive.write(path, 'FlySOC/' + path.relative_to(ROOT).as_posix())
        archive.writestr('FlySOC/RELEASE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity failure')
        for relative, expected in records.items():
            actual = hashlib.sha256(archive.read('FlySOC/' + relative)).hexdigest()
            if actual != expected:
                raise ValueError(f'Archived checksum mismatch: {relative}')
    checksum = sha256(destination)
    destination.with_suffix('.sha256').write_text(f'{checksum}  {destination.name}\n', encoding='utf-8')
    write_json(ROOT / 'results/releases/latest.json', {'path': str(destination), 'sha256': checksum,
                                                      'bytes': destination.stat().st_size, 'files': len(files)})
    print(json.dumps({'zip': str(destination), 'bytes': destination.stat().st_size,
                      'verified_files': len(files), 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    main()
