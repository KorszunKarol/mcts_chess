"""
Unit tests for the HDF5 Replay Buffer.

Tests cover:
1. TalExperience creation and validation
2. HDF5ReplayBuffer buffering and flushing
3. TalDataset iteration and worker splitting
4. Sliding window cleanup
5. Edge cases and error handling
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from src.data.replay_buffer import (
    TalExperience,
    HDF5ReplayBuffer,
    TalDataset,
    create_experience,
    collate_tal_batch,
)
from src.move_mapping import ACTION_SPACE_SIZE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_experience() -> TalExperience:
    """Create a valid sample experience."""
    return TalExperience(
        state=np.random.randn(34, 8, 8).astype(np.float16),
        mcts_policy=np.random.dirichlet(np.ones(ACTION_SPACE_SIZE)).astype(np.float16),
        z_outcome=1.0,
        v_opp_raw=0.5,
    )


@pytest.fixture
def sample_game(sample_experience) -> List[TalExperience]:
    """Create a list of experiences representing one game."""
    return [
        TalExperience(
            state=np.random.randn(34, 8, 8).astype(np.float16),
            mcts_policy=np.random.dirichlet(np.ones(ACTION_SPACE_SIZE)).astype(np.float16),
            z_outcome=np.random.choice([-1.0, 0.0, 1.0]),
            v_opp_raw=np.random.uniform(-1.0, 1.0),
        )
        for _ in range(40)  # Average game length
    ]


# =============================================================================
# TalExperience Tests
# =============================================================================

class TestTalExperience:
    """Tests for TalExperience dataclass."""
    
    def test_valid_creation(self, sample_experience):
        """Test creating a valid experience."""
        assert sample_experience.state.shape == (34, 8, 8)
        assert sample_experience.mcts_policy.shape == (ACTION_SPACE_SIZE,)
        assert isinstance(sample_experience.z_outcome, float)
        assert isinstance(sample_experience.v_opp_raw, float)
    
    def test_invalid_state_shape(self):
        """Test that invalid state shape raises error."""
        with pytest.raises(ValueError, match="State must be"):
            TalExperience(
                state=np.zeros((8, 8, 34), dtype=np.float16),  # Wrong shape (HWC not CHW)
                mcts_policy=np.zeros(ACTION_SPACE_SIZE, dtype=np.float16),
                z_outcome=0.0,
                v_opp_raw=0.0,
            )
    
    def test_invalid_policy_shape(self):
        """Test that invalid policy shape raises error."""
        with pytest.raises(ValueError, match="Policy must be"):
            TalExperience(
                state=np.zeros((34, 8, 8), dtype=np.float16),
                mcts_policy=np.zeros(100, dtype=np.float16),  # Wrong size
                z_outcome=0.0,
                v_opp_raw=0.0,
            )
    
    def test_dtype_preserved(self):
        """Test that float16 dtypes are preserved."""
        exp = TalExperience(
            state=np.zeros((34, 8, 8), dtype=np.float16),
            mcts_policy=np.zeros(ACTION_SPACE_SIZE, dtype=np.float16),
            z_outcome=1.0,
            v_opp_raw=-0.5,
        )
        assert exp.state.dtype == np.float16
        assert exp.mcts_policy.dtype == np.float16


class TestCreateExperience:
    """Tests for the create_experience factory function."""
    
    def test_transposition(self):
        """Test that HWC input is transposed to CHW."""
        state_hwc = np.random.randn(8, 8, 34).astype(np.float32)
        policy = np.random.dirichlet(np.ones(ACTION_SPACE_SIZE))
        
        exp = create_experience(state_hwc, policy, 1.0, 0.5)
        
        assert exp.state.shape == (34, 8, 8)
        # Verify the transposition is correct
        assert np.allclose(exp.state[0], state_hwc[:, :, 0], atol=1e-3)
    
    def test_dtype_conversion(self):
        """Test that arrays are converted to float16."""
        state_hwc = np.random.randn(8, 8, 34).astype(np.float64)
        policy = np.random.randn(ACTION_SPACE_SIZE).astype(np.float64)
        
        exp = create_experience(state_hwc, policy, 1.0, 0.5)
        
        assert exp.state.dtype == np.float16
        assert exp.mcts_policy.dtype == np.float16


# =============================================================================
# HDF5ReplayBuffer Tests
# =============================================================================

class TestHDF5ReplayBuffer:
    """Tests for HDF5ReplayBuffer class."""
    
    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates the data directory."""
        data_dir = temp_dir / "replay"
        buffer = HDF5ReplayBuffer(data_dir)
        assert data_dir.exists()
    
    def test_add_game_buffers_in_memory(self, temp_dir, sample_game):
        """Test that add_game accumulates in memory buffer."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        
        assert buffer.buffer_size == len(sample_game)
        assert buffer.shard_count == 0  # Not flushed yet
    
    def test_flush_writes_shard(self, temp_dir, sample_game):
        """Test that flush writes a shard file."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        shard_path = buffer.flush()
        
        assert shard_path is not None
        assert shard_path.exists()
        assert shard_path.suffix == ".h5"
        assert buffer.buffer_size == 0
    
    def test_auto_flush_on_threshold(self, temp_dir, sample_game):
        """Test automatic flush when threshold is exceeded."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=50)
        
        # Add games until we exceed threshold
        buffer.add_game(sample_game)  # 40 moves
        assert buffer.shard_count == 0
        
        buffer.add_game(sample_game)  # 80 moves total > 50 threshold
        assert buffer.shard_count == 1
        assert buffer.buffer_size == 0
    
    def test_shard_contents(self, temp_dir, sample_game):
        """Test that shard file contains correct data."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        shard_path = buffer.flush()
        
        with h5py.File(shard_path, "r") as f:
            assert f["states"].shape == (len(sample_game), 34, 8, 8)
            assert f["policies"].shape == (len(sample_game), ACTION_SPACE_SIZE)
            assert f["outcomes"].shape == (len(sample_game),)
            assert f["v_opp"].shape == (len(sample_game),)
            assert f.attrs["num_positions"] == len(sample_game)
            assert f.attrs["num_games"] == 1
    
    def test_lzf_compression(self, temp_dir, sample_game):
        """Test that shards use LZF compression."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        shard_path = buffer.flush()
        
        with h5py.File(shard_path, "r") as f:
            assert f["states"].compression == "lzf"
            assert f["policies"].compression == "lzf"
    
    def test_total_positions_count(self, temp_dir, sample_game):
        """Test __len__ returns total positions across shards."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        
        buffer.add_game(sample_game)
        buffer.flush()
        buffer.add_game(sample_game)
        buffer.flush()
        
        assert len(buffer) == len(sample_game) * 2
    
    def test_empty_buffer_flush(self, temp_dir):
        """Test that flushing empty buffer returns None."""
        buffer = HDF5ReplayBuffer(temp_dir)
        result = buffer.flush()
        assert result is None
    
    def test_empty_game_ignored(self, temp_dir):
        """Test that adding empty game list is ignored."""
        buffer = HDF5ReplayBuffer(temp_dir)
        buffer.add_game([])
        assert buffer.buffer_size == 0


class TestSlidingWindow:
    """Tests for sliding window cleanup."""
    
    def test_cleanup_deletes_oldest(self, temp_dir, sample_game):
        """Test that cleanup removes oldest shards."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=10, max_shards=2)
        
        # Create 3 shards
        for _ in range(3):
            buffer.add_game(sample_game[:15])  # Exceeds threshold
            time.sleep(0.01)  # Ensure different timestamps
        buffer.flush()  # Flush any remaining
        
        # Should have max 2 shards
        assert buffer.shard_count <= 2
    
    def test_cleanup_returns_count(self, temp_dir, sample_game):
        """Test that cleanup returns number of deleted shards."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=10, max_shards=1)
        
        buffer.add_game(sample_game[:15])
        buffer.add_game(sample_game[:15])
        
        # Manual cleanup
        deleted = buffer.cleanup_old_shards()
        assert deleted >= 0


# =============================================================================
# TalDataset Tests
# =============================================================================

class TestTalDataset:
    """Tests for TalDataset streaming."""
    
    def test_iteration(self, temp_dir, sample_game):
        """Test basic iteration over dataset."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        dataset = buffer.get_dataset()
        samples = list(dataset)
        
        assert len(samples) == len(sample_game)
    
    def test_sample_structure(self, temp_dir, sample_game):
        """Test that samples have correct structure."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        dataset = buffer.get_dataset()
        sample = next(iter(dataset))
        
        assert "state" in sample
        assert "targets" in sample
        assert sample["state"].shape == (34, 8, 8)
        assert sample["targets"]["policy"].shape == (ACTION_SPACE_SIZE,)
        assert sample["targets"]["val_objective"].shape == ()
        assert sample["targets"]["val_subjective"].shape == ()
    
    def test_tensor_types(self, temp_dir, sample_game):
        """Test that samples are PyTorch tensors."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        dataset = buffer.get_dataset()
        sample = next(iter(dataset))
        
        assert isinstance(sample["state"], torch.Tensor)
        assert isinstance(sample["targets"]["policy"], torch.Tensor)
        assert sample["state"].dtype == torch.float32
    
    def test_dataloader_batching(self, temp_dir, sample_game):
        """Test that DataLoader correctly batches samples."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        loader = buffer.get_dataloader(batch_size=8, num_workers=0)
        batch = next(iter(loader))
        
        assert batch["state"].shape[0] == 8
        assert batch["targets"]["policy"].shape[0] == 8
    
    def test_multiple_shards(self, temp_dir, sample_game):
        """Test iteration over multiple shards."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=10)
        
        # Create multiple shards
        for _ in range(3):
            buffer.add_game(sample_game[:15])
        buffer.flush()
        
        dataset = buffer.get_dataset()
        samples = list(dataset)
        
        # Should have all samples from all shards
        assert len(samples) >= 30
    
    def test_empty_directory_error(self, temp_dir):
        """Test that non-existent directory raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            TalDataset(temp_dir / "nonexistent")
    
    def test_len_estimation(self, temp_dir, sample_game):
        """Test __len__ gives correct total."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        dataset = buffer.get_dataset()
        assert len(dataset) == len(sample_game)


class TestCollateBatch:
    """Tests for custom collate function."""
    
    def test_collate_stacks_tensors(self, temp_dir, sample_game):
        """Test that collate properly stacks samples."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        buffer.add_game(sample_game)
        buffer.flush()
        
        dataset = buffer.get_dataset()
        samples = [next(iter(dataset)) for _ in range(4)]
        
        batch = collate_tal_batch(samples)
        
        assert batch["state"].shape == (4, 34, 8, 8)
        assert batch["targets"]["policy"].shape == (4, ACTION_SPACE_SIZE)
        assert batch["targets"]["val_objective"].shape == (4,)
        assert batch["targets"]["val_subjective"].shape == (4,)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_workflow(self, temp_dir):
        """Test complete workflow: add games, flush, train iteration."""
        buffer = HDF5ReplayBuffer(
            temp_dir,
            flush_threshold=100,
            max_shards=10,
        )
        
        # Simulate self-play: add multiple games
        total_positions = 0
        for _ in range(5):
            game_len = np.random.randint(30, 60)
            game = [
                TalExperience(
                    state=np.random.randn(34, 8, 8).astype(np.float16),
                    mcts_policy=np.random.dirichlet(np.ones(ACTION_SPACE_SIZE)).astype(np.float16),
                    z_outcome=np.random.choice([-1.0, 0.0, 1.0]),
                    v_opp_raw=np.random.uniform(-1.0, 1.0),
                )
                for _ in range(game_len)
            ]
            buffer.add_game(game)
            total_positions += game_len
        
        buffer.flush()
        
        # Verify stats
        stats = buffer.stats()
        assert stats["total_positions"] == total_positions
        
        # Simulate training iteration
        loader = buffer.get_dataloader(batch_size=32, num_workers=0)
        batch_count = 0
        for batch in loader:
            assert batch["state"].shape[0] <= 32
            batch_count += 1
        
        assert batch_count > 0
    
    def test_create_experience_integration(self, temp_dir):
        """Test using create_experience factory with buffer."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=1000)
        
        # Simulate encoder output (HWC format)
        state_hwc = np.random.randn(8, 8, 34).astype(np.float32)
        policy = np.random.dirichlet(np.ones(ACTION_SPACE_SIZE))
        
        exp = create_experience(state_hwc, policy, 1.0, 0.3)
        buffer.add_game([exp])
        buffer.flush()
        
        dataset = buffer.get_dataset()
        sample = next(iter(dataset))
        
        assert sample["state"].shape == (34, 8, 8)


# =============================================================================
# Performance Tests (optional, can be slow)
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance-related tests."""
    
    def test_large_buffer(self, temp_dir):
        """Test buffer with many positions."""
        buffer = HDF5ReplayBuffer(temp_dir, flush_threshold=5000, max_shards=100)
        
        # Add 10k positions
        for _ in range(100):
            game = [
                TalExperience(
                    state=np.random.randn(34, 8, 8).astype(np.float16),
                    mcts_policy=np.random.dirichlet(np.ones(ACTION_SPACE_SIZE)).astype(np.float16),
                    z_outcome=1.0,
                    v_opp_raw=0.0,
                )
                for _ in range(100)
            ]
            buffer.add_game(game)
        
        buffer.flush()
        assert len(buffer) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

