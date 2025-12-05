"""
JAX/Flax model implementations for the Tal-RL pipeline.

This module contains:
- TalModelJAX: Flax port of HybridChessModel
- VictimModel: Frozen model with elevated temperature
- JAX-PyTorch bridge utilities
"""

from src.training_ppo.models.tal_jax import TalModelJAX, ModelOutput
from src.training_ppo.models.victim import VictimModel
from src.training_ppo.models.jax_bridge import torch_to_jax, jax_to_torch

__all__ = [
    "TalModelJAX",
    "ModelOutput",
    "VictimModel",
    "torch_to_jax",
    "jax_to_torch",
]

