"""Regression checks for the rebuilt graph and checkpoint provenance."""
import torch
import pytest

from scripts.make_node_lists import select_nodes
from scripts.prepare_datasets import build_bashapes
from scripts.train_gnn import load_run_model, train_one
from sna.core.provenance import protocol, sha256


def test_rebuilt_bashapes_graph_split_and_features():
    graph = build_bashapes(0, 0)
    x, edges = graph['x'], graph['edge_index']
    masks = torch.stack([graph[f'{split}_mask'] for split in ('train', 'val', 'test')])
    assert x.shape == (700, 10)
    assert edges.shape == (2, 3972)
    assert masks.sum(dim=1).tolist() == [420, 140, 140]
    assert (masks.sum(dim=0) == 1).all()
    assert torch.equal(x[:, 0], torch.bincount(edges[0], minlength=700).float())
    assert torch.equal(x[:, 1], x[:, 0].sqrt())
    assert int(graph['gt_edge_mask'].sum()) == 960
    nodes = select_nodes('bashapes', graph, protocol()['datasets']['bashapes'])
    assert len(nodes) == 140
    assert torch.bincount(graph['y'][nodes]).tolist() == [60, 32, 32, 16]


def test_checkpoint_reloads_and_rejects_a_different_data_hash(tmp_path):
    graph = build_bashapes(0, 0)
    path = tmp_path / 'data.pt'
    torch.save(graph, path)
    info = train_one('bashapes', path, tmp_path, 1000, epochs=2)
    checkpoint = tmp_path / 'gcn_s1000.pt'
    assert info['checkpoint_sha256'] == sha256(checkpoint)
    _, pred, _ = load_run_model(checkpoint, graph, path)
    assert pred.shape == (700,)
    graph['x'][0, 0] += 1
    torch.save(graph, path)
    with pytest.raises(ValueError, match='Data hash mismatch'):
        load_run_model(checkpoint, graph, path)
