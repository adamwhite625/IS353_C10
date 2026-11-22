"""Stage the exact archived T06 data for the final experiment."""
import argparse
from pathlib import Path
import shutil

from sna.core.provenance import execution_metadata, protocol, sha256, write_json


def stage_dataset(dataset, target, source, spec):
    expected = spec['data_sha256']
    if not target.exists():
        if source is None:
            raise FileNotFoundError(f'Provide the original {dataset} file with --{dataset}-data')
        if sha256(source) != expected:
            raise ValueError(f'{dataset}: source data does not match protocol SHA-256')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise ValueError(f'{dataset}: {target} does not match protocol SHA-256. Use a clean data root.')
    write_json(target.with_name('provenance.json'), {
        **execution_metadata(), 'dataset': dataset, 'source': spec['source'],
        'data_sha256': expected, 'dataset_spec': spec,
    })
    print(f'{dataset}: {expected}')


def prepare(data_root, *, bashapes_data=None, cora_data=None):
    specs = protocol()['datasets']
    sources = {'bashapes': bashapes_data, 'cora': cora_data}
    for dataset, source in sources.items():
        stage_dataset(dataset, data_root / dataset / 'data.pt', source, specs[dataset])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('results/data'))
    parser.add_argument('--bashapes-data', type=Path)
    parser.add_argument('--cora-data', type=Path)
    args = parser.parse_args()
    prepare(args.data_root, bashapes_data=args.bashapes_data, cora_data=args.cora_data)


if __name__ == '__main__':
    main()
