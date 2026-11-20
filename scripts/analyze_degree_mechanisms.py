"""BA-Shapes T2/T3: compare motif and 1-hop deletions with fixed and refreshed features."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.utils import degree

from sna.core.provenance import execution_metadata, sha256, write_json
from sna.data.candidate_edges import extract_candidate_edges
from train_gnn import load_run_model


def measure(data_path, node_path, checkpoint_root, output):
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    nodes = json.loads(node_path.read_text())
    if nodes['data_sha256'] != sha256(data_path):
        raise ValueError('Test node list was generated for different data')
    if not data['test_mask'][nodes['nodes']].all():
        raise ValueError('Degree analysis must use test nodes')
    torch.set_num_threads(2)
    edges, x = data['edge_index'], data['x']
    num_nodes = x.shape[0]
    degree_before = degree(edges[0], num_nodes, dtype=x.dtype)
    if not (torch.equal(degree_before, x[:, 0]) and torch.equal(degree_before.sqrt(), x[:, 1])):
        raise ValueError('T3 expects raw degree and square-root degree in columns 0 and 1')
    canonical = edges.min(0).values * num_nodes + edges.max(0).values
    masks = {}
    for node in nodes['nodes']:
        candidates = extract_candidate_edges(edges, node, num_nodes=num_nodes)
        codes = torch.tensor([u * num_nodes + v for u, v in candidates.candidate_pairs],
                             dtype=canonical.dtype)
        motif = torch.isin(canonical, codes) & data['gt_edge_mask']
        direct = (edges == node).any(0) & (edges[0] != edges[1])
        masks[node] = {'motif_2hop': motif, 'one_hop': direct}

    selection_path = checkpoint_root / 'selection.json'
    selection = json.loads(selection_path.read_text())
    rows = []
    checkpoints = {}
    with torch.no_grad():
        for seed in selection['selected_seeds']:
            path = checkpoint_root / f'gcn_s{seed}.pt'
            digest = sha256(path)
            if digest != json.loads(path.with_suffix('.json').read_text())['checkpoint_sha256']:
                raise ValueError(f'Checkpoint hash changed: {path}')
            model, prediction, _ = load_run_model(path, data, data_path)
            checkpoints[path.name] = digest
            original = model(x, edges).softmax(-1)
            for node, interventions in masks.items():
                target = int(prediction[node])
                base = float(original[node, target])
                overlap = canonical[interventions['motif_2hop'] & interventions['one_hop']].unique()
                for name, mask in interventions.items():
                    retained = edges[:, ~mask]
                    degree_after = degree(retained[0], num_nodes, dtype=x.dtype)
                    changed = x.clone()
                    changed[:, 0] = degree_after
                    changed[:, 1] = degree_after.sqrt()
                    fixed = float(model(x, retained)[node].softmax(-1)[target])
                    refreshed = float(model(changed, retained)[node].softmax(-1)[target])
                    rows.append({
                        'dataset': 'bashapes', 'gnn_seed': seed, 'node_id': node,
                        'true_label': int(data['y'][node]), 'target_class': target,
                        'intervention': name, 'removed_pairs': canonical[mask].unique().numel(),
                        'motif_one_hop_overlap': overlap.numel(),
                        'target_degree_before': int(degree_before[node]),
                        'target_degree_after': int(degree_after[node]),
                        'original_probability': base, 'deleted_fixed_probability': fixed,
                        'deleted_refreshed_probability': refreshed,
                        'fid_drop_fixed': base - fixed, 'fid_drop_refreshed': base - refreshed,
                        'delta_refreshed_minus_fixed': fixed - refreshed, 'checkpoint_sha256': digest,
                    })
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / 't2_t3_by_node.csv', index=False)
    summary = metrics.groupby(['gnn_seed', 'intervention']).agg(
        n_nodes=('node_id', 'size'), n_with_edges=('removed_pairs', lambda counts: (counts > 0).sum()),
        mean_removed_pairs=('removed_pairs', 'mean'), mean_fid_drop_fixed=('fid_drop_fixed', 'mean'),
        mean_fid_drop_refreshed=('fid_drop_refreshed', 'mean'),
        mean_delta=('delta_refreshed_minus_fixed', 'mean'),
    )
    summary.to_csv(output / 't2_t3_summary.csv')
    write_json(output / 'manifest.json', {
        **execution_metadata(), 'data_sha256': sha256(data_path), 'nodes_sha256': sha256(node_path),
        'checkpoint_selection_sha256': sha256(selection_path), 'checkpoints': checkpoints,
        'nodes': nodes['nodes'], 'rows': len(rows), 'error_count': 0,
        'outputs': {name: sha256(output / name) for name in ('t2_t3_by_node.csv', 't2_t3_summary.csv')},
        't2': 'remove both directions of all motif pairs in candidate subgraph or all 1-hop pairs',
        't3': 'recompute degree and square-root degree after deletion, then evaluate same checkpoint',
    })
    print(f'T2/T3: {len(rows)} rows across {len(masks)} nodes and {len(checkpoints)} checkpoints')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/data/bashapes/data.pt'))
    parser.add_argument('--nodes', type=Path, default=Path('data/bashapes/test_nodes.json'))
    parser.add_argument('--checkpoints', type=Path, default=Path('results/checkpoints/bashapes'))
    parser.add_argument('--out', type=Path, default=Path('results/analysis/a3_t2_t3'))
    args = parser.parse_args()
    measure(args.data, args.nodes, args.checkpoints, args.out)


if __name__ == '__main__':
    main()
