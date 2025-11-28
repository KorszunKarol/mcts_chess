"""
Data module for Project Tal.

Provides high-performance data storage and streaming for MCTS self-play training.

Key Components:
    - TalExperience: Dataclass representing a single training sample
    - HDF5ReplayBuffer: Sharded buffer for storing self-play experiences
    - TalDataset: PyTorch IterableDataset for streaming from shards
    - create_experience: Factory function for creating experiences with proper format
"""

from src.data.replay_buffer import (
    TalExperience,
    HDF5ReplayBuffer,
    TalDataset,
    create_experience,
    collate_tal_batch,
)

__all__ = [
    "TalExperience",
    "HDF5ReplayBuffer", 
    "TalDataset",
    "create_experience",
    "collate_tal_batch",
]
