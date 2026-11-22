"""Export seed-1000 assets for the legacy integration tests in an isolated directory."""
import argparse
from pathlib import Path
import shutil

import torch

from sna.core.provenance import protocol, sha256, write_json
from train_gnn import load_run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('results/data'))
    parser.add_argument('--checkpoints', type=Path, default=Path('results/checkpoints'))
    parser.add_argument('--out', type=Path, default=Path('results/test-assets'))
    args = parser.parse_args()
    seed = protocol()['fixed_gnn_seed']
    for dataset in protocol()['dataset_order']:
        data_path = args.data_root / dataset / 'data.pt'
        data = torch.load(data_path, weights_only=True, map_location='cpu')
        checkpoint = args.checkpoints / dataset / f'gcn_s{seed}.pt'
        load_run_model(checkpoint, data, data_path)
        dest = args.out / dataset
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_path, dest / 'data.pt')
        shutil.copyfile(checkpoint, dest / 'gcn_ckpt.pt')
        write_json(dest / 'val_nodes.json', {
            'nodes': data['val_mask'].nonzero(as_tuple=True)[0].tolist(),
            'source': 'new_experiment', 'data_sha256': sha256(data_path),
            'checkpoint_sha256': sha256(checkpoint),
        })


if __name__ == '__main__':
    main()
