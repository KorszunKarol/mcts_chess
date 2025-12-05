"""
PPO trainer implementation.

This module provides the PPO algorithm with clipped objective,
value function loss, and entropy bonus for policy optimization.
"""

from src.training_ppo.trainer.ppo import PPOTrainer, PPOConfig

__all__ = [
    "PPOTrainer",
    "PPOConfig",
]

