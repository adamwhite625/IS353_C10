"""Hash experiment inputs and capture their execution context."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

from sna.core.config import PROJECT_ROOT, load_yaml_config


PROTOCOL_PATH = PROJECT_ROOT / 'configs/protocol_final.yaml'


def protocol():
    return load_yaml_config(PROTOCOL_PATH)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def execution_metadata():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    git_status = git('status', '--porcelain')
    return {
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git_status) if git_status is not None else None,
        'utc': datetime.now(timezone.utc).isoformat(),
        'argv': sys.argv,
        'python': sys.version.split()[0],
        'versions': {name: version(name) for name in
                     ('torch', 'torch-geometric', 'numpy', 'pandas', 'scikit-learn')},
        'protocol_sha256': sha256(PROTOCOL_PATH),
        'device': 'cpu',
    }
