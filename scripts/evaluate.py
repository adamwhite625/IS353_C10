"""CLI mỏng của bộ đánh giá v2 (F3). Logic nằm ở ``sna.evaluation.runner_v2``.

Ví dụ (một dataset, toàn bộ lưới):
    PYTHONPATH=src python scripts/evaluate.py --dataset bashapes --scores-root results/scores \
        --out results/eval/bashapes_test
Chia theo seed GNN rồi gộp:
    python scripts/evaluate.py --dataset cora --gnn-seeds 1000 --parts-only ...
    python scripts/evaluate.py --dataset cora --aggregate ...
Tái tạo T06 trên CSV giữa kỳ:
    python scripts/evaluate.py --dataset bashapes --legacy-midterm --scores notebooks/01_midterm/edge_scores.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from sna.evaluation.runner_v2 import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
