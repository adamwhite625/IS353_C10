"""Orchestrate the final protocol, or smoke-test the complete data-to-metric path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp

import pandas as pd
import torch

from make_node_lists import select_nodes
from prepare_datasets import prepare
from train_gnn import load_run_model, train_dataset
from sna.core.config import PROJECT_ROOT, load_yaml_config
from sna.core.provenance import execution_metadata, protocol, sha256, write_json
from sna.core.seed import set_global_seed
from sna.data.candidate_edges import extract_candidate_edges
from sna.evaluation.core import fidelity_selected, ground_truth_metrics, rank_edges, validate_scores
from sna.explainers.pgexplainer import PGExplainerModel
from sna.explainers.registry import get_explainer


def smoke_dataset(dataset, data_root, output, spec):
    seed = spec['fixed_gnn_seed']
    checkpoint_root = output / 'checkpoints'
    train_dataset(dataset, data_root, checkpoint_root, [seed], epochs=2)
    checkpoint = checkpoint_root / dataset / f'gcn_s{seed}.pt'
    checkpoint_hash = sha256(checkpoint)
    data_path = data_root / dataset / 'data.pt'
    data = torch.load(data_path, weights_only=True, map_location='cpu')
    model, prediction, _ = load_run_model(checkpoint, data, data_path)
    node_ids = select_nodes(dataset, data, spec['datasets'][dataset])
    candidates = {}
    for node in node_ids:
        candidate = extract_candidate_edges(data['edge_index'], node, num_nodes=len(prediction))
        if candidate.num_candidate_edges:
            candidates[node] = candidate
        if len(candidates) == 5:
            break
    if len(candidates) != 5:
        raise ValueError(f'{dataset}: fewer than five test nodes with eligible edges')
    chosen = output / dataset / 'test_nodes.json'
    write_json(chosen, {'dataset': dataset, 'nodes': list(candidates), 'data_sha256': sha256(data_path)})

    configs = load_yaml_config(PROJECT_ROOT / f'configs/datasets/{dataset}.yaml')['explainers']
    set_global_seed(0)
    pg = PGExplainerModel(model=model, device=torch.device('cpu'))
    pg.train_on_graph(data, **{**configs['pgexplainer'], 'epochs': 1, 'seed': 0})

    rows = []
    for method in spec['methods']:
        explainer = get_explainer(method, **({'trained_model': pg} if method == 'pgexplainer' else {}))
        for node, candidate in candidates.items():
            res = explainer.explain(model, data, node, int(prediction[node]), candidate,
                                    config={**configs[method], 'epochs': 2}, seed=0)
            if res.status != 'success' or res.edge_ids != candidate.edge_ids:
                raise RuntimeError(f'{dataset}/{method}/{node}: {res.status}: {res.error_message}')
            scores = validate_scores(res.scores, candidate.num_candidate_edges)
            for edge_id, (u, v), score in zip(res.edge_ids, candidate.candidate_pairs, scores):
                rows.append({'dataset': dataset, 'split': 'test', 'arch': 'gcn',
                             'gnn_seed': seed, 'method': method, 'explainer_seed': 0, 'seed': 0,
                             'run_id': f'{dataset}_gcn_g{seed}_{method}_e0_test',
                             'node_id': node, 'edge_id': edge_id, 'u': u, 'v': v,
                             'target_class': int(prediction[node]), 'importance_score': score,
                             'runtime_sec': res.runtime_sec, 'status': 'success',
                             'checkpoint_sha256': checkpoint_hash})
    score_path = output / dataset / 'edge_scores.csv'
    pd.DataFrame(rows).to_csv(score_path, index=False)

    # Read the interchange file back and require matching provenance and complete candidates.
    scores = pd.read_csv(score_path)
    if set(scores['checkpoint_sha256']) != {sha256(checkpoint)}:
        raise ValueError(f'{dataset}: smoke scores use a different checkpoint')
    metrics = []
    for (method, node), group in scores.groupby(['method', 'node_id']):
        candidate = candidates[node]
        group = group.sort_values('edge_id')
        pairs = list(group[['u', 'v']].itertuples(index=False, name=None))
        if pairs != candidate.candidate_pairs or set(group['target_class']) != {int(prediction[node])}:
            raise ValueError(f'{dataset}/{method}/{node}: candidates or predicted class mismatch')
        values = validate_scores(group['importance_score'], len(pairs))
        selected = rank_edges(values, 0)[:spec['main_budget']]
        fid = fidelity_selected(model, data, node, int(prediction[node]), pairs, selected)
        gt = ground_truth_metrics(pairs, values, data['edge_index'], data['gt_edge_mask'], selected)
        metrics.append({'method': method, 'node_id': node, 'gnn_seed': seed, **fid, **gt})
    expected_rows = len(candidates) * len(spec['methods'])
    if len(metrics) != expected_rows:
        raise ValueError(f'{dataset}: expected {expected_rows} smoke metric rows')
    metrics_path = output / dataset / 'metrics.csv'
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    write_json(output / dataset / 'manifest.json', {
        **execution_metadata(), 'mode': 'pn1_smoke', 'data_sha256': sha256(data_path),
        'node_list_sha256': sha256(chosen), 'checkpoint_sha256': checkpoint_hash,
        'source_scores_sha256': sha256(score_path), 'metrics_sha256': sha256(metrics_path),
        'metric_rows': len(metrics), 'error_count': 0,
    })
    print(f'SMOKE PASS {dataset}: {len(candidates)} nodes, {len(metrics)} metric rows', flush=True)


def run_script(name, *args):
    subprocess.run([sys.executable, f'scripts/{name}.py', *map(str, args)],
                   cwd=PROJECT_ROOT, check=True)


def run_full(data_root, spec):
    required = ['run_explainers', 'check_scores', 'evaluate', 'analyze_variance', 'error_analysis']
    missing = [f'scripts/{name}.py' for name in required
               if not (PROJECT_ROOT / f'scripts/{name}.py').is_file()]
    if missing:
        raise SystemExit('Final PN2/PN3 entry points missing: ' + ', '.join(missing))
    selections = {}
    for dataset in spec['dataset_order']:
        run_script('make_node_lists', '--datasets', dataset, '--data-root', data_root)
        selections[dataset] = train_dataset(dataset, data_root, Path('results/checkpoints'),
                                            spec['gnn_seeds'])
    for dataset in spec['dataset_order']:
        # Selection preserves declared order including any replacement of the reference seed.
        scores = Path('results/scores') / dataset
        evaluation = Path('results/eval') / f'{dataset}_test'
        run_script('run_explainers', '--dataset', dataset, '--split', 'test',
                   '--data-dir', data_root, '--nodes-file', Path('data') / dataset / 'test_nodes.json',
                   '--gnn-seeds', *selections[dataset], '--protocol-saliency',
                   '--explainer-seeds', *spec['explainer_seeds'], '--methods', *spec['methods'])
        run_script('check_scores', scores, '--data-dir', data_root,
                   '--expected-nodes-file', Path('data') / dataset / 'test_nodes.json')
        run_script('evaluate', '--dataset', dataset, '--data-root', data_root,
                   '--scores', scores, '--out', evaluation)
        run_script('analyze_variance', '--eval', evaluation)
    run_script('analyze_degree_mechanisms', '--data', data_root / 'bashapes/data.pt')
    run_script('error_analysis', '--dataset', 'bashapes', '--eval', 'results/eval/bashapes_test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--bashapes-data', type=Path)
    parser.add_argument('--cora-data', type=Path)
    args = parser.parse_args()
    spec = protocol()
    if args.smoke:
        smoke_root = PROJECT_ROOT / 'results/smoke'
        smoke_root.mkdir(parents=True, exist_ok=True)
        output = Path(mkdtemp(prefix='run-', dir=smoke_root))
        data_root = args.data_root or output / 'data'
        local = PROJECT_ROOT / 'results/data'
        sources = {}
        for ds in spec['dataset_order']:
            path = local / ds / 'data.pt'
            sources[f'{ds}_data'] = getattr(args, f'{ds}_data') or (path if path.exists() else None)
        prepare(data_root, **sources)
        for dataset in spec['dataset_order']:
            smoke_dataset(dataset, data_root, output, spec)
        print(f'Smoke results: {output}')
    else:
        data_root = args.data_root or PROJECT_ROOT / 'results/data'
        prepare(data_root, bashapes_data=args.bashapes_data, cora_data=args.cora_data)
        run_full(data_root.resolve(), spec)


if __name__ == '__main__':
    main()
