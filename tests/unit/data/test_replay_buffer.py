"""
Tests for the Tal HDF5 replay buffer and dataset utilities.

These tests align with the current API in src.data.replay_buffer and ensure
basic functionality works end-to-end.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from src.data.replay_buffer import (
    TalExperience,
    HDF5ReplayBuffer,
    TalDataset,
    create_experience,
    collate_tal_batch,
)
from src.move_mapping import ACTION_SPACE_SIZE


def make_experience(z_outcome: float = 1.0, v_opp: float = 0.25) -> TalExperience:
    """Create a minimal TalExperience with deterministic contents."""
    policy = np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE
    state = np.zeros((8, 8, 34), dtype=np.float32)
    return create_experience(
        state_hwc=state,
        mcts_policy=policy,
        z_outcome=z_outcome,
        v_opp_raw=v_opp,
    )


def make_game(length: int) -> list[TalExperience]:
    """Create a list of TalExperience objects representing one game."""
    return [
        make_experience(z_outcome=float((i % 3) - 1), v_opp=0.1 * i)
        for i in range(length)
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def replay_buffer(temp_dir):
    """Create an HDF5ReplayBuffer instance with a low flush threshold."""
    return HDF5ReplayBuffer(
        data_dir=temp_dir,
        flush_threshold=4,
        max_shards=5,
    )


def test_create_experience_transposes_and_casts():
    """TalExperience should transpose HWC -> CHW and cast to float16."""
    exp = make_experience(z_outcome=1.0, v_opp=0.5)

    assert isinstance(exp, TalExperience)
    assert exp.state.shape == (34, 8, 8)
    assert exp.state.dtype == np.float16
    assert exp.mcts_policy.shape == (ACTION_SPACE_SIZE,)
    assert exp.mcts_policy.dtype == np.float16
    assert exp.z_outcome == 1.0
    assert exp.v_opp_raw == 0.5


def test_add_and_flush_creates_shard(replay_buffer, temp_dir):
    """Flushing should create a shard file with correctly shaped datasets."""
    replay_buffer.add_game(make_game(4))

    shards = list(temp_dir.glob("shard_*.h5"))
    assert len(shards) == 1

    with h5py.File(shards[0], "r") as f:
        assert f["states"].shape == (4, 34, 8, 8)
        assert f["policies"].shape == (4, ACTION_SPACE_SIZE)
        assert f["outcomes"].shape == (4,)
        assert f["v_opp"].shape == (4,)


def test_dataset_iteration_and_collate(temp_dir):
    """TalDataset should yield samples and collate into batched tensors."""
    buffer = HDF5ReplayBuffer(data_dir=temp_dir, flush_threshold=2, max_shards=5)
    buffer.add_game(make_game(2))

    dataset = TalDataset(temp_dir)
    sample = next(iter(dataset))

    assert sample["state"].shape == (34, 8, 8)
    assert sample["targets"]["policy"].shape == (ACTION_SPACE_SIZE,)
    assert sample["targets"]["val_objective"].ndim == 0
    assert sample["targets"]["val_subjective"].ndim == 0

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_tal_batch, num_workers=0)
    batch = next(iter(loader))

    assert batch["state"].shape == (2, 34, 8, 8)
    assert batch["targets"]["policy"].shape == (2, ACTION_SPACE_SIZE)
    assert batch["targets"]["val_objective"].shape == (2,)
    assert batch["targets"]["val_subjective"].shape == (2,)


def test_sliding_window_cleanup_respects_max(temp_dir):
    """cleanup_old_shards should enforce the configured max_shards limit."""
    buffer = HDF5ReplayBuffer(data_dir=temp_dir, flush_threshold=1, max_shards=2)

    for _ in range(4):
        buffer.add_game(make_game(1))

    assert buffer.shard_count == 2


