"""
Rollout buffer for PPO training.

This module provides on-policy storage for collecting experience
during rollouts, computing GAE advantages, and generating minibatches.
"""

from src.training_ppo.storage.rollout_buffer import RolloutBuffer, Batch

__all__ = [
    "RolloutBuffer",
    "Batch",
]

