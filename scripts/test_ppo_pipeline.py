#!/usr/bin/env python3
"""
Test script for validating the Tal-RL PPO pipeline.

This script runs a minimal training loop with reduced parameters
to verify all components work together correctly.

Usage:
    python scripts/test_ppo_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration loading."""
    logger.info("Testing config...")
    
    from src.training_ppo.config import PPOTalConfig, PPOConfig, TalRewardConfig
    
    # Default config
    config = PPOTalConfig()
    assert config.env.num_envs == 4096
    assert config.mcts.num_simulations == 50
    
    # Nested configs
    ppo = config.ppo
    assert ppo.lr == 3e-4
    assert ppo.clip_range == 0.2
    
    logger.info("  Config: OK")
    return config


def test_rollout_buffer():
    """Test rollout buffer functionality."""
    logger.info("Testing rollout buffer...")
    
    from src.training_ppo.storage.rollout_buffer import RolloutBuffer
    
    num_steps = 8
    num_envs = 16
    obs_shape = (34, 8, 8)
    device = "cpu"
    
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_shape=obs_shape,
        device=device,
    )
    
    # Add some data
    for step in range(num_steps):
        buffer.add(
            obs=torch.randn(num_envs, *obs_shape),
            action=torch.randint(0, 100, (num_envs,)),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
            value=torch.randn(num_envs),
            log_prob=torch.randn(num_envs),
            q_truth=torch.randn(num_envs),
            victim_entropy=torch.randn(num_envs),
        )
    
    assert buffer.is_full
    
    # Compute returns
    last_value = torch.randn(num_envs)
    buffer.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
    
    assert buffer.ready
    
    # Get batches
    batch_count = 0
    for batch in buffer.get_batches(batch_size=32):
        assert batch.obs.shape[0] <= 32
        batch_count += 1
    
    assert batch_count > 0
    
    # Reset
    buffer.reset()
    assert not buffer.ready
    assert buffer.step == 0
    
    logger.info("  Rollout buffer: OK")


def test_ppo_trainer():
    """Test PPO trainer."""
    logger.info("Testing PPO trainer...")
    
    from src.training_ppo.trainer.ppo import PPOTrainer, PPOConfig
    from src.training_ppo.storage.rollout_buffer import RolloutBuffer
    
    # Simple test model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(34 * 8 * 8, 256)
            self.value_head = torch.nn.Linear(256, 3)
            self.policy_head = torch.nn.Linear(256, 4672)
        
        def forward(self, x):
            x = x.flatten(1)
            x = torch.relu(self.fc(x))
            value = torch.softmax(self.value_head(x), dim=-1)
            policy = self.policy_head(x)
            return value, policy
    
    model = SimpleModel()
    config = PPOConfig(ppo_epochs=1, minibatch_size=16)
    trainer = PPOTrainer(model, config, device="cpu")
    
    # Create buffer with data
    buffer = RolloutBuffer(
        num_steps=8,
        num_envs=16,
        obs_shape=(34, 8, 8),
        device="cpu",
    )
    
    for _ in range(8):
        buffer.add(
            obs=torch.randn(16, 34, 8, 8),
            action=torch.randint(0, 4672, (16,)),
            reward=torch.randn(16),
            done=torch.zeros(16),
            value=torch.randn(16),
            log_prob=torch.randn(16),
        )
    
    buffer.compute_returns_and_advantages(torch.randn(16))
    
    # Run update
    metrics = trainer.update(buffer)
    
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    
    logger.info("  PPO trainer: OK")


def test_reward_engine():
    """Test Tal reward engine."""
    logger.info("Testing reward engine...")
    
    # Import without JAX for this test
    from src.training_ppo.rewards.normalizer import RunningMeanStd
    
    rms = RunningMeanStd()
    
    # Update with some data
    for _ in range(10):
        data = np.random.randn(100)
        rms.update(data)
    
    assert rms.count > 0
    assert abs(rms.mean) < 1.0  # Should be near 0 for standard normal
    
    # Normalize
    test_data = np.array([0.0, 1.0, -1.0])
    normalized = rms.normalize(test_data)
    assert normalized.shape == test_data.shape
    
    logger.info("  Reward engine: OK")


def test_metrics_logger():
    """Test metrics logger (without WandB)."""
    logger.info("Testing metrics logger...")
    
    from src.training_ppo.metrics.logger import TalMetricsLogger, MetricAggregator
    
    # Test aggregator
    agg = MetricAggregator()
    agg.add({"loss": 0.5, "reward": 1.0})
    agg.add({"loss": 0.3, "reward": 0.8})
    
    stats = agg.get_stats()
    assert "loss" in stats
    assert stats["loss"]["count"] == 2
    assert abs(stats["loss"]["mean"] - 0.4) < 0.01
    
    # Test logger with TensorBoard
    logger_obj = TalMetricsLogger()
    logger_obj.log_step({"reward": 1.0}, num_envs=1)
    logger_obj.log_iteration(0, {"policy_loss": 0.1})
    logger_obj.finish()
    
    logger.info("  Metrics logger: OK")


def test_full_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.training_ppo import PPOConfig, TalRewardConfig
        from src.training_ppo.storage import RolloutBuffer, Batch
        from src.training_ppo.trainer import PPOTrainer
        from src.training_ppo.metrics import TalMetricsLogger
        from src.training_ppo.rewards import TalRewardEngine, RunningMeanStd
        logger.info("  Core imports: OK")
    except ImportError as e:
        logger.warning(f"  Core imports: SKIP ({e})")
    
    # JAX imports (may not be available)
    try:
        import jax
        import jax.numpy as jnp
        from src.training_ppo.env import VectorizedChessEnv
        from src.training_ppo.models import TalModelJAX
        from src.training_ppo.mcts import BatchedMCTS
        logger.info("  JAX imports: OK")
    except ImportError as e:
        logger.warning(f"  JAX imports: SKIP ({e})")


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Tal-RL PPO Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Config", test_config),
        ("Rollout Buffer", test_rollout_buffer),
        ("PPO Trainer", test_ppo_trainer),
        ("Reward Engine", test_reward_engine),
        ("Metrics Logger", test_metrics_logger),
        ("Full Imports", test_full_imports),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"  {name}: FAILED - {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

