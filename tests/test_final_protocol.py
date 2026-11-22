"""Regression checks for archived inputs and checkpoint provenance."""
import pytest
import torch

from scripts.prepare_datasets import stage_dataset
from scripts.train_gnn import load_run_model, train_one
from sna.core.provenance import sha256


def test_archived_dataset_requires_protocol_hash(tmp_path):
    source = tmp_path / 'source.pt'
    source.write_bytes(b'archived data')
    spec = {'source': 'archived_t06', 'data_sha256': sha256(source)}
    target = tmp_path / 'staged' / 'data.pt'
    stage_dataset('bashapes', target, source, spec)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b'different data')
    with pytest.raises(ValueError, match='does not match protocol SHA-256'):
        stage_dataset('bashapes', target, source, spec)


def test_checkpoint_reloads_and_rejects_a_different_data_hash(tmp_path):
    graph = {
        'x': torch.arange(12, dtype=torch.float32).unsqueeze(1),
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'y': torch.arange(12) % 4,
        'train_mask': torch.arange(12) < 4,
        'val_mask': (torch.arange(12) >= 4) & (torch.arange(12) < 8),
        'test_mask': torch.arange(12) >= 8,
    }
    path = tmp_path / 'data.pt'
    torch.save(graph, path)
    info = train_one('bashapes', path, tmp_path, 1000, epochs=2)
    checkpoint = tmp_path / 'gcn_s1000.pt'
    assert info['checkpoint_sha256'] == sha256(checkpoint)
    _, pred, _ = load_run_model(checkpoint, graph, path)
    assert pred.shape == (12,)
    graph['x'][0, 0] += 1
    torch.save(graph, path)
    with pytest.raises(ValueError, match='Data hash mismatch'):
        load_run_model(checkpoint, graph, path)
