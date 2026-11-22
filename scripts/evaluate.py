"""Evaluate final-protocol score CSVs (F3). Stops on any provenance mismatch.

Example:
  python scripts/evaluate.py --dataset bashapes --scores results/scores/bashapes \
      --out results/eval/bashapes_test
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from sna.core.config import PROJECT_ROOT
from sna.core.provenance import execution_metadata, protocol
from sna.evaluation.runner_v2 import HashMismatchError, evaluate


def main(argv=None):
    spec = protocol()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(spec['datasets']))
    parser.add_argument('--scores', type=Path, required=True, help='results/scores/{dataset}')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data',
                        help='directory holding {dataset}/data.pt')
    parser.add_argument('--checkpoints-dir', type=Path, default=PROJECT_ROOT / 'results/checkpoints')
    parser.add_argument('--nodes-file', type=Path, default=None,
                        help='default: data/{dataset}/test_nodes.json')
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--budgets', type=int, nargs='+', default=spec['budgets'])
    parser.add_argument('--tie-seeds', type=int, nargs='+', default=spec['tie_sensitivity_seeds'])
    parser.add_argument('--baseline-seeds', type=int, nargs='+', default=spec['baseline_repeat_seeds'])
    parser.add_argument('--max-nodes', type=int, default=None, help='first N nodes only (smoke/dev)')
    parser.add_argument('--no-ensemble', action='store_true', help='skip D1 seed-ensemble runs')
    args = parser.parse_args(argv)

    torch.set_num_threads(spec['torch_threads'])
    nodes_file = args.nodes_file or PROJECT_ROOT / 'data' / args.dataset / f'{args.split}_nodes.json'
    started = time.perf_counter()
    try:
        manifest = evaluate(
            args.dataset, args.data_root / args.dataset / 'data.pt', args.scores,
            args.checkpoints_dir, nodes_file, args.out, budgets=args.budgets,
            tie_seeds=args.tie_seeds, baseline_seeds=args.baseline_seeds,
            expected_data_sha256=spec['datasets'][args.dataset]['data_sha256'],
            max_nodes=args.max_nodes, ensemble=not args.no_ensemble, split=args.split,
            metadata=execution_metadata(), explainer_seeds=spec['explainer_seeds'],
            primary_gnn_seed=spec['fixed_gnn_seed'])
    except HashMismatchError as exc:
        print(f'HASH MISMATCH, evaluation stopped: {exc}', file=sys.stderr)
        return 2
    print(f"[{manifest['status']}] {args.dataset}: {manifest['n_nodes']} nodes, "
          f"{len(manifest['runs'])} runs, rows={manifest['rows']} "
          f"({time.perf_counter() - started:.0f}s) -> {args.out}")
    for w in manifest['warnings'][:10]:
        print(f'  [WARN] {w}')
    if manifest['missing_protocol_runs']:
        print(f"  [WARN] {len(manifest['missing_protocol_runs'])} protocol runs missing, "
              f"e.g. {manifest['missing_protocol_runs'][:5]}")
    if manifest['error_count']:
        print(f"  {manifest['error_count']} errors, see {args.out / 'errors.csv'}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
