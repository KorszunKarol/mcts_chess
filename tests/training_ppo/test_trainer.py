import time

import pytest
import torch
import torch.nn as nn

from src.training_ppo.storage.rollout_buffer import RolloutBuffer
from src.training_ppo.trainer.ppo import PPOConfig, PPOTrainer


class TinyPolicy(nn.Module):
    """Small network to make PPO overfitting tests fast and deterministic."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 3)

    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        hidden = self.body(x)

        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)
        value_probs = torch.softmax(value_logits, dim=-1)

        return value_probs, policy_logits


def _build_buffer(
    device: str,
    obs_dim: int = 4,
    num_steps: int = 4,
    num_envs: int = 2,
    action_index: int = 0,
) -> RolloutBuffer:
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_shape=(obs_dim,),
        device=device,
    )

    for step in range(num_steps):
        obs = torch.ones((num_envs, obs_dim), device=device) * step
        actions = torch.full((num_envs,), action_index, dtype=torch.long, device=device)
        reward = torch.ones(num_envs, device=device)
        done = torch.zeros(num_envs, device=device)
        value = torch.zeros(num_envs, device=device)
        log_prob = torch.zeros(num_envs, device=device)

        buffer.add(
            obs=obs,
            action=actions,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )

    last_value = torch.zeros(num_envs, device=device)
    buffer.compute_returns_and_advantages(
        last_value=last_value,
        gamma=0.9,
        gae_lambda=0.95,
    )
    return buffer


def test_overfit_single_batch_learning_curves_drop():
    device = "cpu"
    torch.manual_seed(0)

    model = TinyPolicy(obs_dim=4, action_dim=4).to(device)
    trainer = PPOTrainer(
        model=model,
        config=PPOConfig(
            lr=5e-3,
            clip_range=0.2,
            entropy_coef=0.0,
            value_coef=0.5,
            max_grad_norm=0.5,
            ppo_epochs=3,
            minibatch_size=4,
        ),
        device=device,
    )

    buffer = _build_buffer(
        device=device,
        obs_dim=4,
        num_steps=4,
        num_envs=1,
        action_index=0,
    )

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(15):
        metrics = trainer.update(buffer)
        policy_losses.append(metrics["policy_loss"])
        value_losses.append(metrics["value_loss"])
        entropies.append(metrics["entropy"])

    assert policy_losses[-1] < policy_losses[0]
    assert value_losses[-1] <= value_losses[0]
    assert entropies[-1] < entropies[0]


def test_two_update_cycle_writes_logs_and_stays_stable(tmp_path):
    device = "cpu"
    torch.manual_seed(1)

    model = TinyPolicy(obs_dim=4, action_dim=3).to(device)
    trainer = PPOTrainer(
        model=model,
        config=PPOConfig(
            lr=1e-3,
            clip_range=0.2,
            entropy_coef=0.0,
            value_coef=0.5,
            max_grad_norm=0.5,
            ppo_epochs=2,
            minibatch_size=4,
        ),
        device=device,
    )

    buffer = _build_buffer(
        device=device,
        obs_dim=4,
        num_steps=4,
        num_envs=2,
        action_index=1,
    )

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tmp_path)

    cuda_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    durations = []

    for step in range(2):
        start = time.perf_counter()
        metrics = trainer.update(buffer)
        durations.append(time.perf_counter() - start)

        for key, value in metrics.items():
            writer.add_scalar(f"ppo/{key}", value, step)

    writer.flush()
    writer.close()

    assert all(duration < 5.0 for duration in durations)
    assert list(tmp_path.iterdir()), "TensorBoard logs were not written"

    if cuda_before is not None:
        cuda_after = torch.cuda.memory_allocated()
        assert cuda_after <= cuda_before + 10 * 1024 * 1024  # within 10 MB


