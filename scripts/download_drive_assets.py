"""Locate research assets and verify them after downloading from the team Drive."""
import argparse
import json
from pathlib import Path
import sys

from sna.core.config import PROJECT_ROOT
from sna.core.provenance import sha256


DRIVE_LINK = "https://drive.google.com/drive/folders/1Mfc4SMsLTqO8jTGRTMF5FjYYo39A1lTY?usp=sharing"
T06 = PROJECT_ROOT / 'notebooks/05_duc_kha_evaluation/results/manifest.json'


def verify(root, expected):
    failures = 0
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            print(f'MISSING {name}')
            failures += 1
        else:
            actual = sha256(path)
            status = 'OK' if actual == digest else 'HASH MISMATCH'
            print(f'{status} {name} {actual}')
            failures += actual != digest
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--manifest', type=Path, default=T06,
                        help='JSON containing a filename-to-SHA-256 "assets" mapping')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if not manifest.get('assets'):
        parser.error('Manifest has no assets')
    print(f'Team Drive: {DRIVE_LINK}\nAsset root: {args.root}')
    return int(verify(args.root, manifest['assets']) != 0)


if __name__ == '__main__':
    sys.exit(main())
