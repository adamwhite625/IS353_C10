"""Prepare reconstructed BA-Shapes and verify the original Cora data."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.utils import degree, sort_edge_index

from sna.core.provenance import execution_metadata, protocol, sha256, write_json
from sna.core.seed import set_global_seed


def build_bashapes(graph_seed, split_seed):
    set_global_seed(graph_seed)
    graph = ExplainerDataset(BAGraph(300, 5), HouseMotif(), num_motifs=80)[0]
    edges, motif_mask = sort_edge_index(graph.edge_index, graph.edge_mask.bool())
    non_loop = edges[0] != edges[1]
    edges, motif_mask = edges[:, non_loop], motif_mask[non_loop]
    n = len(graph.y)
    # The two degree features are unnormalized, with eight zero padding columns.
    graph_degree = degree(edges[0], n, dtype=torch.float32)
    x = torch.zeros(n, 10)
    x[:, 0] = graph_degree
    x[:, 1] = graph_degree.sqrt()
    nodes = np.arange(n)
    train, heldout = train_test_split(nodes, test_size=0.4, stratify=graph.y.numpy(),
                                      random_state=split_seed)
    val, test = train_test_split(heldout, test_size=0.5, stratify=graph.y[heldout].numpy(),
                                 random_state=split_seed)
    masks = {}
    for name, subset in zip(('train', 'val', 'test'), (train, val, test)):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[subset] = True
        masks[f'{name}_mask'] = mask
    canonical = edges.min(dim=0).values * n + edges.max(dim=0).values
    _, pair_id = torch.unique(canonical, sorted=True, return_inverse=True)
    stats = {'name': 'bashapes', 'nodes': n, 'edge_entries': edges.shape[1],
             'features': 10, 'classes': 4, 'train': len(train), 'val': len(val),
             'test': len(test), 'undirected': True, 'self_loops_removed': int((~non_loop).sum())}
    return {'x': x, 'edge_index': edges, 'y': graph.y, 'gt_edge_mask': motif_mask,
            'pair_id': pair_id, 'undirected': True, 'stats': stats, **masks}


def prepare(data_root, cora_data=None):
    cfg = protocol()['datasets']
    ba = data_root / 'bashapes'
    ba.mkdir(parents=True, exist_ok=True)
    if not (ba / 'data.pt').exists():
        spec = cfg['bashapes']
        data = build_bashapes(spec['graph_seed'], spec['split_seed'])
        torch.save(data, ba / 'data.pt')
        write_json(ba / 'provenance.json', {
            **execution_metadata(), 'dataset': 'bashapes', 'source': 'reconstructed',
            'data_sha256': sha256(ba / 'data.pt'), 'dataset_spec': spec, 'stats': data['stats'],
            'generator': 'PyG ExplainerDataset(BAGraph(300, 5), HouseMotif(), num_motifs=80)',
            'features': 'raw degree, square-root degree, eight zeros',
            'split': 'two-stage node-stratified 60/20/20 by label',
        })
    info = ba / 'provenance.json'
    if not info.exists() or json.loads(info.read_text())['data_sha256'] != sha256(ba / 'data.pt'):
        raise ValueError(f'Reconstructed BA-Shapes provenance mismatch: {ba}')

    cora = data_root / 'cora'
    cora.mkdir(parents=True, exist_ok=True)
    target = cora / 'data.pt'
    expected = cfg['cora']['data_sha256']
    if not target.exists():
        if cora_data is None:
            raise FileNotFoundError('Provide the archived Cora file with --cora-data')
        if sha256(cora_data) != expected:
            raise ValueError('Cora file does not match the T06 manifest')
        shutil.copyfile(cora_data, target)
    if sha256(target) != expected:
        raise ValueError(f'Cora file does not match the T06 manifest: {target}')
    write_json(cora / 'provenance.json', {
        **execution_metadata(), 'dataset': 'cora', 'source': 'archived_t06',
        'data_sha256': expected,
    })
    for ds in ('bashapes', 'cora'):
        print(f'{ds}: {sha256(data_root / ds / "data.pt")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('results/data'))
    parser.add_argument('--cora-data', type=Path)
    args = parser.parse_args()
    prepare(args.data_root, args.cora_data)


if __name__ == '__main__':
    main()
