"""FB-2 analysis (variance.py, scripts/analyze_variance.py)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import json
import numpy as np
import pandas as pd
import pytest
from sna.evaluation.variance import (between, cross_gnn, prediction_changes, table_a2_1,
                                     table_a2_2, verdicts, within)


def row(kind, node, ma, mb, ga, gb, ea, eb, kappa, va='single', vb='single', same=True):
    return dict(dataset='toy', kind=kind, node_id=node, budget=5, method_a=ma, variant_a=va,
                gnn_seed_a=ga, explainer_seed_a=ea, method_b=mb, variant_b=vb, gnn_seed_b=gb,
                explainer_seed_b=eb, same_pred=same, broadcast=False, kappa=kappa,
                jaccard=kappa, kendall_tau_b=kappa)


@pytest.fixture
def toy():
    rows = []
    rng = np.random.default_rng(0)
    for node in range(30):
        for g in (1000, 1001):
            # A is stable (~0.9), B unstable (~0.2); they disagree (~0.1).
            rows.append(row('within', node, 'gnnexplainer', 'gnnexplainer', g, g, 0, 1, 0.9 + rng.normal(0, .02)))
            rows.append(row('within', node, 'pgexplainer', 'pgexplainer', g, g, 0, 1, 0.2 + rng.normal(0, .02)))
            for e in (0, 1):
                rows.append(row('between', node, 'gnnexplainer', 'pgexplainer', g, g, e, e, 0.1 + rng.normal(0, .02)))
            rows.append(row('between_ensemble', node, 'gnnexplainer', 'pgexplainer', g, g, -1, -1,
                            0.15 + rng.normal(0, .02), 'ensemble', 'ensemble'))
        for e in (0, 1):
            rows.append(row('cross_gnn', node, 'gnnexplainer', 'gnnexplainer', 1000, 1001, e, e, 0.5))
            rows.append(row('cross_gnn', node, 'pgexplainer', 'pgexplainer', 1000, 1001, e, e, 0.2))
        # A changed prediction pair must be ignored by I_gnn.
        rows.append(row('cross_gnn', node, 'gnnexplainer', 'gnnexplainer', 1000, 1001, 2, 2, -1.0, same=False))
        rows.append(row('ref_random', node, 'random', 'random', 1000, 1000, 0, 1, 0.0, 'baseline', 'baseline'))
    return pd.DataFrame(rows)


def test_definitions_average_within_node_and_respect_scope(toy):
    w_all = within(toy, 'gnnexplainer', 'kappa')
    w_fixed = within(toy, 'gnnexplainer', 'kappa', gnn_seeds=[1000])
    assert len(w_all) == 30 and w_all.mean() == pytest.approx(0.9, abs=0.01)
    one = toy[(toy.kind == 'within') & (toy.method_a == 'gnnexplainer') & (toy.node_id == 3)]
    assert w_all.loc[3] == pytest.approx(one.kappa.mean())
    assert w_fixed.loc[3] == pytest.approx(one[one.gnn_seed_a == 1000].kappa.iloc[0])
    assert cross_gnn(toy, 'gnnexplainer', 'kappa').eq(0.5).all()   # same_pred=False row dropped
    # Pair order must not matter.
    assert between(toy, 'pgexplainer', 'gnnexplainer', 'kappa').equals(
        between(toy, 'gnnexplainer', 'pgexplainer', 'kappa'))


def test_tables_and_rules(toy):
    t1 = table_a2_1(toy, 'toy', 'kappa', 5, 'all', None)
    t2 = table_a2_2(toy, 'toy', 'kappa', 5, 'all', None)
    df2 = pd.DataFrame(t2)
    da = df2[df2.quantity == 'delta_a'].iloc[0]
    db = df2[df2.quantity == 'delta_b'].iloc[0]
    assert da['mean'] == pytest.approx(0.8, abs=0.02) and da.ci_lo > 0
    assert db['mean'] == pytest.approx(0.1, abs=0.02) and db.ci_lo > 0
    assert df2[df2.quantity == 'delta_avg'].iloc[0]['mean'] == pytest.approx(0.05, abs=0.02)
    assert 'p_holm' in da and da.p_holm >= da.p
    df1 = pd.DataFrame(t1)
    gap = df1[(df1.method == 'gnnexplainer') & (df1.quantity == 'delta_within_minus_gnn')].iloc[0]
    assert gap['mean'] == pytest.approx(0.4, abs=0.02)
    text = ' '.join(verdicts(t1, t2))
    assert 'bất đồng thực sự' in text and 'gộp seed (D1) giúp ích' in text
    assert 'gnnexplainer: I_gnn thấp hơn rõ I_within' in text


def test_no_genuine_disagreement_when_within_is_as_low_as_between():
    rows = []
    for node in range(30):
        rows.append(row('within', node, 'gnnexplainer', 'gnnexplainer', 1000, 1000, 0, 1, 0.1))
        rows.append(row('within', node, 'pgexplainer', 'pgexplainer', 1000, 1000, 0, 1, 0.9))
        rows.append(row('between', node, 'gnnexplainer', 'pgexplainer', 1000, 1000, 0, 0, 0.1))
    ag = pd.DataFrame(rows)
    text = ' '.join(verdicts([], table_a2_2(ag, 'toy', 'kappa', 5, 'fixed', [1000])))
    assert 'chưa đủ bằng chứng' in text and 'gnnexplainer' in text


def test_prediction_changes():
    m = pd.DataFrame({'gnn_seed': [1000, 1001, 1002] * 2, 'node_id': [1, 1, 1, 2, 2, 2],
                      'target_class': [0, 0, 0, 1, 2, 1]})
    out = prediction_changes(m)
    assert out['node_change_rate'] == pytest.approx(0.5)
    assert out['pair_change_rate'] == pytest.approx(2 / 6)


def test_cli_end_to_end(tmp_path):
    from test_runner_v2 import CKPTS, DATA, DS, NODES, write_run
    import analyze_variance
    from sna.core.provenance import sha256
    from sna.evaluation.runner_v2 import evaluate
    scores = tmp_path / 'scores'
    for g in (1000, 1001):
        for method in ('gnnexplainer', 'pgexplainer'):
            for e in (0, 1):
                write_run(scores, g, method, e, NODES)
        write_run(scores, g, 'saliency', 0, NODES)
    nodes_file = tmp_path / 'nodes.json'
    nodes_file.write_text(json.dumps({'nodes': NODES, 'data_sha256': sha256(DATA)}))
    evaluate(DS, DATA, scores, CKPTS, nodes_file, tmp_path / 'eval', budgets=[5],
             tie_seeds=[0], baseline_seeds=[0, 1], metadata={}, explainer_seeds=[0, 1])
    out = tmp_path / 'a2'
    assert analyze_variance.main(['--eval', str(tmp_path / 'eval'), '--out', str(out),
                                  '--budgets', '5']) == 0
    for name in ('a2_1.csv', 'a2_2.csv', 'a2_3.csv', 'fig_a2.png', 'report.md', 'manifest.json'):
        assert (out / name).is_file()
    t2 = pd.read_csv(out / 'a2_2.csv')
    assert set(t2.scope) == {'fixed', 'all'}
    assert len(t2[(t2.metric == 'kappa') & (t2.quantity == 'D_between')]) == 2 * 3
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['inputs_sha256']['agreement.csv'] == sha256(tmp_path / 'eval' / 'agreement.csv')
    assert 'Bảng A2-1' in (out / 'report.md').read_text(encoding='utf-8')
