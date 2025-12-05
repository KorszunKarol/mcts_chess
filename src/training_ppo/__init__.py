"""
Tal-RL PPO Training Pipeline.

This module implements an asynchronous PPO training pipeline for learning
cognitive asymmetry in chess. The key components are:

- VectorizedChessEnv: JAX/pgx-based parallel game simulation (4096 games)
- TalModelJAX: Flax port of the HybridChessModel for mctx compatibility
- BatchedMCTS: Vectorized MCTS using DeepMind's mctx library
- TalRewardEngine: Computes cognitive asymmetry rewards
- PPOTrainer: Standard PPO with clipped objective

The training objective maximizes:
    J(s') = V_φ(s') + λ·[E_user(s') + γ·H_opp(s')]

Where:
    - V_φ(s') is the oracle (MCTS) evaluation
    - E_user is user ease (policy sharpness)
    - H_opp is opponent entropy (confusion)

Usage:
    # Training
    python scripts/train_ppo.py --config configs/ppo_tal.yaml
    
    # Testing
    python scripts/test_ppo_pipeline.py
"""

from src.training_ppo.config import (
    PPOTalConfig,
    PPOConfig,
    TalRewardConfig,
    EnvConfig,
    AgentConfig,
    VictimConfig,
    MCTSConfig,
    TrainingConfig,
)

__all__ = [
    # Main config
    "PPOTalConfig",
    # Sub-configs
    "PPOConfig",
    "TalRewardConfig", 
    "EnvConfig",
    "AgentConfig",
    "VictimConfig",
    "MCTSConfig",
    "TrainingConfig",
]

