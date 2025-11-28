"""
Sharded HDF5 Replay Buffer for Project Tal.

This module provides high-performance data storage and streaming for MCTS self-play
experiences. Key design decisions:

1. **Sharding**: Games are buffered in memory and flushed to HDF5 shards when
   threshold is reached (~5000 positions). This avoids the "Small File Problem"
   where thousands of tiny files would cripple I/O throughput.

2. **Type Optimization**: States and policies stored as float16 to reduce disk
   usage by 2x while maintaining sufficient precision for training.

3. **Pre-transposition**: States stored in PyTorch NCHW format (34, 8, 8) to
   avoid millions of runtime transpositions during training.

4. **LZF Compression**: Fastest decompression for training throughput.

5. **Worker-safe Dataset**: IterableDataset that correctly splits shards across
   DataLoader workers for parallel loading.

Usage:
    buffer = HDF5ReplayBuffer(Path("./replay_data"))
    
    # During self-play
    for game in self_play_games:
        buffer.add_game(game_experiences)
    buffer.flush()  # Ensure remaining buffer is written
    
    # During training
    dataset = buffer.get_dataset()
    loader = DataLoader(dataset, batch_size=256, num_workers=4)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
import uuid
import time
import logging
import os

import numpy as np
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader

from src.move_mapping import ACTION_SPACE_SIZE

log = logging.getLogger(__name__)


# =============================================================================
# Data Schema
# =============================================================================

@dataclass
class TalExperience:
    """
    A single training sample from MCTS self-play.
    
    This represents one position from a game, with the MCTS-derived targets
    and the opponent model's evaluation for asymmetric learning.
    
    Attributes:
        state: Board encoding in PyTorch NCHW format (34, 8, 8), float16.
               Pre-transposed from encoder's HWC output for training efficiency.
        mcts_policy: Normalized MCTS visit counts over action space (4672,), float16.
        z_outcome: Game result from perspective of player to move: +1 win, 0 draw, -1 loss.
        v_opp_raw: Opponent model's V_subj evaluation of this position.
                   Used to train the subjective value head to predict opponent confusion.
    """
    state: np.ndarray        # (34, 8, 8) float16 - Pre-transposed for PyTorch NCHW
    mcts_policy: np.ndarray  # (4672,) float16 - normalized visit counts
    z_outcome: float         # float32 - Game result: +1, 0, -1
    v_opp_raw: float         # float32 - Opponent's V_subj during self-play
    
    def __post_init__(self):
        """Validate shapes and types."""
        if self.state.shape != (34, 8, 8):
            raise ValueError(f"State must be (34, 8, 8), got {self.state.shape}")
        if self.mcts_policy.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(
                f"Policy must be ({ACTION_SPACE_SIZE},), got {self.mcts_policy.shape}"
            )


def create_experience(
    state_hwc: np.ndarray,
    mcts_policy: np.ndarray,
    z_outcome: float,
    v_opp_raw: float,
) -> TalExperience:
    """
    Factory function to create TalExperience with proper transposition.
    
    Use this when your encoder outputs HWC format (8, 8, 34).
    The state will be transposed to CHW format (34, 8, 8) and converted to float16.
    
    Args:
        state_hwc: Board encoding from encoder, shape (8, 8, 34)
        mcts_policy: Normalized MCTS visit counts, shape (4672,)
        z_outcome: Game result from player's perspective
        v_opp_raw: Opponent's subjective evaluation
        
    Returns:
        TalExperience with properly formatted arrays
    """
    # Transpose HWC -> CHW and convert to float16
    state_chw = np.transpose(state_hwc, (2, 0, 1)).astype(np.float16)
    policy_f16 = mcts_policy.astype(np.float16)
    
    return TalExperience(
        state=state_chw,
        mcts_policy=policy_f16,
        z_outcome=float(z_outcome),
        v_opp_raw=float(v_opp_raw),
    )


# =============================================================================
# HDF5 Replay Buffer (Sharded)
# =============================================================================

class HDF5ReplayBuffer:
    """
    High-performance replay buffer using sharded HDF5 storage.
    
    Games are accumulated in memory and flushed to disk as shards when
    the buffer exceeds the flush threshold. This avoids the overhead
    of managing thousands of small files.
    
    Attributes:
        data_dir: Directory for storing shard files
        flush_threshold: Number of positions to buffer before flushing
        max_shards: Maximum number of shards to keep (sliding window)
        
    Example:
        buffer = HDF5ReplayBuffer(Path("./data/replay"))
        
        # Add games from self-play
        for game_experiences in self_play():
            buffer.add_game(game_experiences)
            
        # Force flush at end of session
        buffer.flush()
        
        # Get dataset for training
        dataset = buffer.get_dataset()
        loader = DataLoader(dataset, batch_size=256, num_workers=4)
    """
    
    SHARD_PATTERN = "shard_*.h5"
    
    def __init__(
        self,
        data_dir: Path,
        flush_threshold: int = 5000,
        max_shards: int = 500,
    ):
        """
        Initialize the replay buffer.
        
        Args:
            data_dir: Directory to store HDF5 shard files
            flush_threshold: Flush to disk when buffer exceeds this many positions
            max_shards: Maximum shards to keep; oldest are deleted (sliding window)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.flush_threshold = flush_threshold
        self.max_shards = max_shards
        
        # In-memory buffer for accumulating experiences
        self._buffer: List[TalExperience] = []
        self._games_in_buffer: int = 0
        
        log.info(
            f"HDF5ReplayBuffer initialized: dir={self.data_dir}, "
            f"flush_threshold={flush_threshold}, max_shards={max_shards}"
        )
    
    def add_game(self, experiences: List[TalExperience]) -> None:
        """
        Add a completed game's experiences to the buffer.
        
        The experiences are kept in memory until flush_threshold is exceeded,
        at which point they are written to disk as a shard.
        
        Args:
            experiences: List of TalExperience from one game (typically 40-80 moves)
        """
        if not experiences:
            return
            
        self._buffer.extend(experiences)
        self._games_in_buffer += 1
        
        # Check if we should flush
        if len(self._buffer) >= self.flush_threshold:
            self.flush()
    
    def flush(self) -> Optional[Path]:
        """
        Write buffered experiences to a new shard file.
        
        Returns:
            Path to the created shard file, or None if buffer was empty
        """
        if not self._buffer:
            return None
        
        # Stack experiences into numpy arrays
        states = np.stack([exp.state for exp in self._buffer], axis=0)
        policies = np.stack([exp.mcts_policy for exp in self._buffer], axis=0)
        outcomes = np.array([exp.z_outcome for exp in self._buffer], dtype=np.float32)
        v_opps = np.array([exp.v_opp_raw for exp in self._buffer], dtype=np.float32)
        
        # Generate unique shard filename
        timestamp = int(time.time() * 1000)
        unique_id = uuid.uuid4().hex[:8]
        shard_name = f"shard_{timestamp}_{unique_id}.h5"
        shard_path = self.data_dir / shard_name
        temp_path = shard_path.with_suffix(".h5.tmp")
        
        # Write to temp file first (atomic write pattern)
        try:
            with h5py.File(temp_path, "w") as f:
                # Use LZF compression - fastest decompression for training
                f.create_dataset(
                    "states",
                    data=states,
                    dtype="float16",
                    compression="lzf",
                    chunks=(min(256, len(states)), 34, 8, 8),
                )
                f.create_dataset(
                    "policies",
                    data=policies,
                    dtype="float16",
                    compression="lzf",
                    chunks=(min(256, len(policies)), ACTION_SPACE_SIZE),
                )
                f.create_dataset(
                    "outcomes",
                    data=outcomes,
                    dtype="float32",
                    compression="lzf",
                )
                f.create_dataset(
                    "v_opp",
                    data=v_opps,
                    dtype="float32",
                    compression="lzf",
                )
                
                # Store metadata
                f.attrs["num_positions"] = len(self._buffer)
                f.attrs["num_games"] = self._games_in_buffer
                f.attrs["timestamp"] = timestamp
                
            # Atomic rename
            temp_path.rename(shard_path)
            
            log.info(
                f"Flushed shard: {shard_name} "
                f"({len(self._buffer)} positions, {self._games_in_buffer} games)"
            )
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to write shard: {e}") from e
        
        # Clear buffer
        num_flushed = len(self._buffer)
        self._buffer.clear()
        self._games_in_buffer = 0
        
        # Clean up old shards if needed
        self.cleanup_old_shards()
        
        return shard_path
    
    def cleanup_old_shards(self) -> int:
        """
        Delete oldest shards to maintain sliding window.
        
        Shards are sorted by timestamp (embedded in filename) and the
        oldest are deleted until we're under max_shards.
        
        Returns:
            Number of shards deleted
        """
        shards = self._get_shard_files()
        
        if len(shards) <= self.max_shards:
            return 0
        
        # Sort by modification time (oldest first)
        shards_with_time = [(s, s.stat().st_mtime) for s in shards]
        shards_with_time.sort(key=lambda x: x[1])
        
        # Delete oldest until under limit
        to_delete = len(shards) - self.max_shards
        deleted = 0
        
        for shard_path, _ in shards_with_time[:to_delete]:
            try:
                shard_path.unlink()
                deleted += 1
                log.debug(f"Deleted old shard: {shard_path.name}")
            except OSError as e:
                log.warning(f"Failed to delete shard {shard_path}: {e}")
        
        if deleted > 0:
            log.info(f"Cleaned up {deleted} old shards (sliding window)")
        
        return deleted
    
    def _get_shard_files(self) -> List[Path]:
        """Get all valid shard files in data directory."""
        return list(self.data_dir.glob(self.SHARD_PATTERN))
    
    def get_dataset(self) -> "TalDataset":
        """
        Create a streaming dataset for training.
        
        Returns:
            TalDataset instance that can be used with PyTorch DataLoader
        """
        return TalDataset(self.data_dir)
    
    def get_dataloader(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """
        Create a DataLoader ready for training.
        
        Args:
            batch_size: Samples per batch
            num_workers: Parallel data loading workers
            prefetch_factor: Batches to prefetch per worker
            
        Returns:
            Configured DataLoader instance
        """
        dataset = self.get_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )
    
    def __len__(self) -> int:
        """Total positions across all shards (excluding in-memory buffer)."""
        total = 0
        for shard_path in self._get_shard_files():
            try:
                with h5py.File(shard_path, "r") as f:
                    total += f.attrs.get("num_positions", len(f["states"]))
            except Exception:
                pass  # Skip corrupted shards
        return total
    
    @property
    def buffer_size(self) -> int:
        """Number of positions currently in memory buffer."""
        return len(self._buffer)
    
    @property
    def shard_count(self) -> int:
        """Number of shard files on disk."""
        return len(self._get_shard_files())
    
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "data_dir": str(self.data_dir),
            "shard_count": self.shard_count,
            "total_positions": len(self),
            "buffer_positions": self.buffer_size,
            "buffer_games": self._games_in_buffer,
            "flush_threshold": self.flush_threshold,
            "max_shards": self.max_shards,
        }


# =============================================================================
# PyTorch Dataset
# =============================================================================

class TalDataset(IterableDataset):
    """
    Streaming dataset that reads from HDF5 shards.
    
    This dataset is designed for efficient multi-worker data loading:
    - Shards are split across DataLoader workers
    - Each worker processes its subset of shards independently
    - Positions within each shard are shuffled
    
    The dataset yields dictionaries with structure:
        {
            "state": Tensor(34, 8, 8),
            "targets": {
                "policy": Tensor(4672,),
                "val_objective": Tensor(scalar),
                "val_subjective": Tensor(scalar),
            }
        }
    """
    
    SHARD_PATTERN = "shard_*.h5"
    
    def __init__(self, data_dir: Path, shuffle_shards: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing HDF5 shard files
            shuffle_shards: Whether to shuffle shard order each epoch
        """
        self.data_dir = Path(data_dir)
        self.shuffle_shards = shuffle_shards
        
        # Validate directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
    
    def _get_shard_files(self) -> List[Path]:
        """Get all shard files, optionally shuffled."""
        shards = list(self.data_dir.glob(self.SHARD_PATTERN))
        if self.shuffle_shards:
            np.random.shuffle(shards)
        return shards
    
    def _get_worker_shards(self, worker_info) -> List[Path]:
        """
        Split shards across DataLoader workers.
        
        Each worker gets a disjoint subset of shards to process,
        ensuring no duplicate data and balanced load.
        """
        all_shards = self._get_shard_files()
        
        if worker_info is None:
            # Single-process loading
            return all_shards
        
        # Multi-process: split shards by worker id
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        
        # Round-robin assignment
        worker_shards = [
            shard for i, shard in enumerate(all_shards) 
            if i % num_workers == worker_id
        ]
        
        return worker_shards
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over all positions in worker's assigned shards.
        
        Positions within each shard are shuffled for better training.
        """
        worker_info = torch.utils.data.get_worker_info()
        my_shards = self._get_worker_shards(worker_info)
        
        for shard_path in my_shards:
            yield from self._iter_shard(shard_path)
    
    def _iter_shard(self, shard_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over all positions in a single shard, shuffled."""
        try:
            with h5py.File(shard_path, "r") as f:
                num_positions = len(f["states"])
                
                # Shuffle indices for this shard
                indices = np.random.permutation(num_positions)
                
                for idx in indices:
                    yield self._get_item(f, int(idx))
                    
        except Exception as e:
            log.warning(f"Error reading shard {shard_path}: {e}")
            return  # Skip corrupted shards
    
    def _get_item(self, f: h5py.File, idx: int) -> Dict[str, torch.Tensor]:
        """Extract a single training sample from an open HDF5 file."""
        state = torch.from_numpy(f["states"][idx].astype(np.float32))
        policy = torch.from_numpy(f["policies"][idx].astype(np.float32))
        outcome = torch.tensor(f["outcomes"][idx], dtype=torch.float32)
        v_opp = torch.tensor(f["v_opp"][idx], dtype=torch.float32)
        
        return {
            "state": state,
            "targets": {
                "policy": policy,
                "val_objective": outcome,
                "val_subjective": v_opp,
            }
        }
    
    def __len__(self) -> int:
        """
        Total positions across all shards.
        
        Note: For IterableDataset, __len__ is optional and some DataLoader
        features may not work correctly with it. Use for estimation only.
        """
        total = 0
        for shard_path in self.data_dir.glob(self.SHARD_PATTERN):
            try:
                with h5py.File(shard_path, "r") as f:
                    total += len(f["states"])
            except Exception:
                pass
        return total


# =============================================================================
# Utility Functions
# =============================================================================

def collate_tal_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for TalDataset.
    
    This is the default collation if not using the built-in DataLoader collation.
    Stacks samples into batched tensors.
    
    Args:
        batch: List of sample dictionaries from TalDataset
        
    Returns:
        Batched dictionary with same structure
    """
    states = torch.stack([sample["state"] for sample in batch])
    policies = torch.stack([sample["targets"]["policy"] for sample in batch])
    outcomes = torch.stack([sample["targets"]["val_objective"] for sample in batch])
    v_opps = torch.stack([sample["targets"]["val_subjective"] for sample in batch])
    
    return {
        "state": states,
        "targets": {
            "policy": policies,
            "val_objective": outcomes,
            "val_subjective": v_opps,
        }
    }


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("HDF5 Replay Buffer Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = HDF5ReplayBuffer(
            Path(tmpdir),
            flush_threshold=100,  # Low threshold for demo
            max_shards=5,
        )
        
        # Generate fake game data
        print("\n1. Adding games to buffer...")
        for game_idx in range(10):
            game_length = np.random.randint(30, 60)
            experiences = []
            for move_idx in range(game_length):
                exp = TalExperience(
                    state=np.random.randn(34, 8, 8).astype(np.float16),
                    mcts_policy=np.random.dirichlet(np.ones(ACTION_SPACE_SIZE)).astype(np.float16),
                    z_outcome=np.random.choice([-1.0, 0.0, 1.0]),
                    v_opp_raw=np.random.uniform(-1.0, 1.0),
                )
                experiences.append(exp)
            buffer.add_game(experiences)
            print(f"  Game {game_idx + 1}: {game_length} moves, buffer size: {buffer.buffer_size}")
        
        # Flush remaining
        buffer.flush()
        
        print(f"\n2. Buffer stats: {buffer.stats()}")
        
        # Test dataset
        print("\n3. Testing TalDataset...")
        dataset = buffer.get_dataset()
        print(f"  Total positions: {len(dataset)}")
        
        # Iterate a few samples
        print("\n4. Sample iteration:")
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            print(f"  Sample {i}: state={sample['state'].shape}, "
                  f"policy={sample['targets']['policy'].shape}, "
                  f"outcome={sample['targets']['val_objective'].item():.2f}")
        
        # Test DataLoader
        print("\n5. Testing DataLoader...")
        loader = buffer.get_dataloader(batch_size=32, num_workers=0)
        for batch in loader:
            print(f"  Batch: state={batch['state'].shape}, "
                  f"policy={batch['targets']['policy'].shape}")
            break
        
        print("\n✓ All tests passed!")

