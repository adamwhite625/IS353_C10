"""Checks for FB-3 summaries that distinguish seed repeats from test nodes."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from error_analysis import METHODS, configuration_tables, node_tables, winners


def test_tied_winner_sets_require_disjoint_methods_to_count_as_conflict():
    part = pd.DataFrame({"method": METHODS[:3], "fid_drop": [.4, .4, .1],
                         "roc_auc": [.5, .8, .7]})
    assert winners(part)["conflict"] is False
    part.loc[1, "roc_auc"] = .6
    assert winners(part)["conflict"] is True
    part["roc_auc"] = np.nan
    assert winners(part)["conflict"] is None


def test_configuration_means_average_repeats_within_node_and_preserve_undefined_auc():
    rows = []
    for method in METHODS:
        for node in (0, 1):
            for tie in (0, 1):
                rows.append(dict(gnn_seed=1000, explainer_seed=0, method=method, node_id=node,
                                 tie_seed=tie, fid_drop=.1 * (node + tie),
                                 roc_auc=.75 if node else np.nan))
    cfg = {"statistics": {"bootstrap": {"repeats": 100}}}
    metrics, conflicts = configuration_tables(pd.DataFrame(rows), [1000], [0], cfg)
    drop = metrics.query("method == 'pgexplainer' and metric == 'fid_drop'").iloc[0]
    roc = metrics.query("method == 'pgexplainer' and metric == 'roc_auc'").iloc[0]
    assert np.isclose(drop["mean"], .1)
    assert (drop.n_valid, roc.n_valid, roc.n_total) == (2, 1, 2)
    assert len(conflicts) == 3


def test_quadrants_include_threshold_equality_and_preserve_missing_auroc():
    rows = pd.DataFrame([dict(gnn_seed=1000, node_id=i, checkpoint_sha256="hash",
                              method="pgexplainer", true_label=1, target_class=1,
                              fid_drop=.10, fid_inv=.2, roc_auc=auc,
                              precision_at_5=.4, one_hop_fraction=.4, tied_fraction=0,
                              gap_5_6=.2) for i, auc in enumerate((.75, np.nan))])
    cfg = {"error_analysis": {"quadrant_thresholds": [[.10, .75]]}}
    _, conflicts, quadrants, _ = node_tables(rows, cfg)
    assert quadrants.quadrant.tolist() == ["Q1", "undefined"]
    assert pd.isna(conflicts.iloc[1].conflict)
