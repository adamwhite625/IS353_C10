"""Create and validate the fixed test node lists before scoring."""
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch

from sna.core.provenance import protocol, verify_protocol_data, write_json


def select_nodes(dataset, data, spec):
    test = data['test_mask'].nonzero(as_tuple=True)[0]
    if spec['test_nodes'] == 'all':
        chosen = test.tolist()
    else:
        _, chosen = train_test_split(test.tolist(), test_size=spec['test_nodes'],
                                     stratify=data['y'][test].numpy(),
                                     random_state=spec['sample_seed'])
        chosen = sorted(chosen)
    if len(set(chosen)) != len(chosen) or not chosen:
        raise ValueError(f'{dataset}: empty or duplicate test nodes')
    if not data['test_mask'][chosen].all() or (data['train_mask'] | data['val_mask'])[chosen].any():
        raise ValueError(f'{dataset}: test selection overlaps train or validation')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['bashapes', 'cora'],
                        default=protocol()['dataset_order'])
    parser.add_argument('--data-root', type=Path, default=Path('results/data'))
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    for ds in args.datasets:
        path = args.data_root / ds / 'data.pt'
        data_hash = verify_protocol_data(ds, path)
        data = torch.load(path, weights_only=True, map_location='cpu')
        nodes = select_nodes(ds, data, protocol()['datasets'][ds])
        counts = torch.bincount(data['y'][nodes], minlength=data['stats']['classes']).tolist()
        write_json(args.output / ds / 'test_nodes.json', {
            'dataset': ds, 'split': 'test', 'data_sha256': data_hash,
            'nodes': nodes, 'class_counts': counts,
        })
        print(f'{ds}: {len(nodes)} node; class counts {counts}')


if __name__ == '__main__':
    main()
