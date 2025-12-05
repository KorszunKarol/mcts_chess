"""
Rollout Buffer for PPO training.

Stores on-policy experience during rollouts and provides:
- GAE (Generalized Advantage Estimation) computation
- Minibatch generation for PPO updates
- Automatic buffer reset after each update

This is an on-policy buffer: data is collected, used for one
PPO update, then discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, NamedTuple, Dict, Any, List

import torch
import numpy as np
from src.utils import sentinel

logger = logging.getLogger(__name__)


class Batch(NamedTuple):
    """A minibatch of experience for PPO update."""
    obs: torch.Tensor          # (B, C, H, W) observations
    actions: torch.Tensor      # (B,) action indices
    old_log_probs: torch.Tensor  # (B,) log prob of actions under old policy
    advantages: torch.Tensor   # (B,) GAE advantages
    returns: torch.Tensor      # (B,) discounted returns (for value loss)
    values: torch.Tensor       # (B,) old value predictions
    
    # Additional Tal-specific data
    q_truth: torch.Tensor      # (B,) MCTS Q-values
    victim_entropy: torch.Tensor  # (B,) victim policy entropy
    
    # Style metrics (optional, may be None in older code paths)
    material_imbalance: Optional[torch.Tensor] = None  # (B,) white - black material
    chaos_index: Optional[torch.Tensor] = None         # (B,) sound moves available
    agent_suicide: Optional[torch.Tensor] = None       # (B,) suicide move flags


@dataclass
class EpisodeStats:
    """Statistics for a completed episode."""
    outcome: float        # +1 win, -1 loss, 0 draw
    length: int           # number of moves
    final_material: float  # material imbalance at game end
    total_value_gap: float  # sum of value gaps during game
    suicide_count: int    # number of suicide moves


@dataclass
class RolloutBuffer:
    """
    PPO-style on-policy rollout buffer.
    
    Stores experience from parallel environments during rollout,
    computes advantages using GAE, and generates minibatches.
    
    Example:
        buffer = RolloutBuffer(num_steps=128, num_envs=4096, obs_shape=(34, 8, 8))
        
        # Rollout phase
        for step in range(num_steps):
            action, log_prob, value = agent(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, done, value, log_prob, q_truth, victim_entropy)
            obs = next_obs
        
        # Compute advantages
        buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)
        
        # PPO update
        for batch in buffer.get_batches(minibatch_size):
            loss = ppo.update(batch)
        
        # Reset for next rollout
        buffer.reset()
    """
    num_steps: int
    num_envs: int
    obs_shape: tuple
    device: str = "cuda"
    
    def __post_init__(self):
        """Initialize storage tensors."""
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs, *self.obs_shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.long,
            device=self.device,
        )
        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.dones = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.log_probs = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Tal-specific storage
        self.q_truth = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.victim_entropy = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Computed after rollout
        self.advantages = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.returns = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Style tracking tensors (Tal personality verification)
        self.material_imbalance = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.chaos_index = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.agent_suicide = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.value_gap = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Episode tracking (per-environment state)
        # These track the current episode in each env
        self._episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_value_gap_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_suicide_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Completed episode statistics (accumulated during rollout)
        self.completed_episodes: List[EpisodeStats] = []
        
        self.step = 0
        self.ready = False
        
        logger.info(
            f"RolloutBuffer initialized: {self.num_steps} steps, "
            f"{self.num_envs} envs, {self.num_steps * self.num_envs} total samples"
        )
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        q_truth: Optional[torch.Tensor] = None,
        victim_entropy: Optional[torch.Tensor] = None,
        material_imbalance: Optional[torch.Tensor] = None,
        chaos_index: Optional[torch.Tensor] = None,
        agent_suicide: Optional[torch.Tensor] = None,
        value_gap: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add a single timestep of experience.
        
        Args:
            obs: (B, C, H, W) observations.
            action: (B,) action indices.
            reward: (B,) rewards.
            done: (B,) done flags.
            value: (B,) value predictions.
            log_prob: (B,) log probs of actions.
            q_truth: (B,) optional MCTS Q-values.
            victim_entropy: (B,) optional victim entropy.
            material_imbalance: (B,) optional material difference (white - black).
            chaos_index: (B,) optional opponent sound move count.
            agent_suicide: (B,) optional suicide move flags.
            value_gap: (B,) optional Q_truth - V_victim.
        """
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full. Call reset() or compute_returns.")

        batch = obs.shape[0]
        if batch != self.num_envs:
            raise ValueError(
                f"Obs batch size {batch} does not match buffer.num_envs={self.num_envs}"
            )

        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        if sentinel.enabled:
            sentinel.log(f"Buffer step {self.step}/{self.num_steps}")
            sentinel.log_tensor("obs", obs)
            sentinel.log_tensor("actions", action)
            sentinel.log_tensor("reward", reward)
            sentinel.log_tensor("done", done)
        
        if q_truth is not None:
            self.q_truth[self.step] = q_truth
        if victim_entropy is not None:
            self.victim_entropy[self.step] = victim_entropy
        
        # Style metrics
        if material_imbalance is not None:
            self.material_imbalance[self.step] = material_imbalance
        if chaos_index is not None:
            self.chaos_index[self.step] = chaos_index
        if agent_suicide is not None:
            self.agent_suicide[self.step] = agent_suicide
        if value_gap is not None:
            self.value_gap[self.step] = value_gap
        
        # Update episode tracking
        self._update_episode_tracking(done, material_imbalance, value_gap, agent_suicide, reward)
        
        self.step += 1
    
    def _update_episode_tracking(
        self,
        done: torch.Tensor,
        material_imbalance: Optional[torch.Tensor],
        value_gap: Optional[torch.Tensor],
        agent_suicide: Optional[torch.Tensor],
        reward: torch.Tensor,
    ) -> None:
        """
        Update per-episode tracking and record completed episodes.
        
        Called after each step to accumulate episode statistics and
        record them when episodes complete.
        """
        # Increment episode lengths for all envs
        self._episode_lengths += 1
        
        # Accumulate value gap
        if value_gap is not None:
            self._episode_value_gap_sum += value_gap
        
        # Accumulate suicide count
        if agent_suicide is not None:
            self._episode_suicide_count += agent_suicide.long()
        
        # Check for completed episodes
        done_mask = done.bool()
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=True)[0]
            
            for idx in done_indices:
                idx = idx.item()
                
                # Determine outcome from reward (assuming terminal reward)
                # +1 for win, -1 for loss, 0 for draw
                outcome = float(reward[idx].item())
                if outcome > 0.5:
                    outcome = 1.0
                elif outcome < -0.5:
                    outcome = -1.0
                else:
                    outcome = 0.0
                
                # Record episode stats
                stats = EpisodeStats(
                    outcome=outcome,
                    length=int(self._episode_lengths[idx].item()),
                    final_material=float(material_imbalance[idx].item()) if material_imbalance is not None else 0.0,
                    total_value_gap=float(self._episode_value_gap_sum[idx].item()),
                    suicide_count=int(self._episode_suicide_count[idx].item()),
                )
                self.completed_episodes.append(stats)
                
                # Reset tracking for this env
                self._episode_lengths[idx] = 0
                self._episode_value_gap_sum[idx] = 0.0
                self._episode_suicide_count[idx] = 0
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute returns and GAE advantages.
        
        Uses Generalized Advantage Estimation (GAE) for lower
        variance advantage estimates.
        
        GAE: A_t = δ_t + γλ·A_{t+1}
        where δ_t = r_t + γ·V_{t+1} - V_t
        
        Args:
            last_value: (B,) value estimate for state after last step.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        gae = torch.zeros(self.num_envs, device=self.device)
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_value = last_value
                next_done = torch.zeros(self.num_envs, device=self.device)
            else:
                next_value = self.values[step + 1]
                next_done = self.dones[step + 1]
            
            # TD error
            delta = (
                self.rewards[step] 
                + gamma * next_value * (1 - next_done)
                - self.values[step]
            )
            
            # GAE
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            self.advantages[step] = gae
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
        self.ready = True
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Batch]:
        """
        Generate minibatches for PPO update.
        
        Args:
            batch_size: Size of each minibatch.
            shuffle: Whether to shuffle data.
            
        Yields:
            Batch namedtuples.
        """
        if not self.ready:
            raise RuntimeError("Call compute_returns_and_advantages() first.")
        
        total_samples = self.num_steps * self.num_envs
        
        # Flatten all data
        flat_obs = self.obs.reshape(-1, *self.obs_shape)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)
        flat_q_truth = self.q_truth.reshape(-1)
        flat_victim_entropy = self.victim_entropy.reshape(-1)
        
        # Normalize advantages (important for stable training)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # Generate indices
        if shuffle:
            indices = torch.randperm(total_samples, device=self.device)
        else:
            indices = torch.arange(total_samples, device=self.device)
        
        # Flatten style metrics
        flat_material = self.material_imbalance.reshape(-1)
        flat_chaos = self.chaos_index.reshape(-1)
        flat_suicide = self.agent_suicide.reshape(-1)
        
        # Yield batches
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            yield Batch(
                obs=flat_obs[batch_indices],
                actions=flat_actions[batch_indices],
                old_log_probs=flat_log_probs[batch_indices],
                advantages=flat_advantages[batch_indices],
                returns=flat_returns[batch_indices],
                values=flat_values[batch_indices],
                q_truth=flat_q_truth[batch_indices],
                victim_entropy=flat_victim_entropy[batch_indices],
                material_imbalance=flat_material[batch_indices],
                chaos_index=flat_chaos[batch_indices],
                agent_suicide=flat_suicide[batch_indices],
            )
    
    def reset(self) -> None:
        """Reset buffer for next rollout."""
        self.step = 0
        self.ready = False
        
        # Zero out tensors (optional, for cleanliness)
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.q_truth.zero_()
        self.victim_entropy.zero_()
        self.advantages.zero_()
        self.returns.zero_()
        
        # Reset style tracking
        self.material_imbalance.zero_()
        self.chaos_index.zero_()
        self.agent_suicide.zero_()
        self.value_gap.zero_()
        
        # Note: Do NOT reset episode tracking here as episodes may span rollouts
        # completed_episodes is cleared separately via get_episode_statistics()
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.step >= self.num_steps
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about stored data."""
        stats = {
            "reward_mean": self.rewards[:self.step].mean().item(),
            "reward_std": self.rewards[:self.step].std().item(),
            "value_mean": self.values[:self.step].mean().item(),
            "q_truth_mean": self.q_truth[:self.step].mean().item(),
            "victim_entropy_mean": self.victim_entropy[:self.step].mean().item(),
            "done_rate": self.dones[:self.step].mean().item(),
        }
        
        # Add style metrics if available
        if self.step > 0:
            stats["material_imbalance_mean"] = self.material_imbalance[:self.step].mean().item()
            stats["chaos_index_mean"] = self.chaos_index[:self.step].mean().item()
            stats["agent_suicide_rate"] = self.agent_suicide[:self.step].mean().item()
            stats["value_gap_mean"] = self.value_gap[:self.step].mean().item()
        
        return stats
    
    def get_episode_statistics(self, clear: bool = True) -> Dict[str, float]:
        """
        Get statistics from completed episodes.
        
        Args:
            clear: Whether to clear completed episodes after computing stats.
            
        Returns:
            Dictionary with episode-level metrics.
        """
        if not self.completed_episodes:
            return {}
        
        episodes = self.completed_episodes
        
        # Compute aggregates
        outcomes = [e.outcome for e in episodes]
        lengths = [e.length for e in episodes]
        
        # Win/loss/draw rates
        wins = sum(1 for o in outcomes if o > 0)
        losses = sum(1 for o in outcomes if o < 0)
        draws = sum(1 for o in outcomes if o == 0)
        total = len(outcomes)
        
        # Material in winning games (Tal metric: lower is more Tal-like)
        winning_materials = [e.final_material for e in episodes if e.outcome > 0]
        
        # Suicide rate across episodes
        total_moves = sum(e.length for e in episodes)
        total_suicides = sum(e.suicide_count for e in episodes)
        
        stats = {
            "win_rate": wins / total if total > 0 else 0.0,
            "loss_rate": losses / total if total > 0 else 0.0,
            "draw_rate": draws / total if total > 0 else 0.0,
            "game_length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "game_length_min": min(lengths) if lengths else 0.0,
            "game_length_max": max(lengths) if lengths else 0.0,
            "episode_count": total,
        }
        
        # Material imbalance in wins (Tal wants this low or negative)
        if winning_materials:
            stats["material_in_wins_mean"] = sum(winning_materials) / len(winning_materials)
        
        # Agent suicide rate (safety check)
        if total_moves > 0:
            stats["suicide_rate_episode"] = total_suicides / total_moves
        
        if clear:
            self.completed_episodes.clear()
        
        return stats


def create_buffer(
    num_steps: int,
    num_envs: int,
    obs_shape: tuple,
    device: str = "cuda",
) -> RolloutBuffer:
    """
    Factory function to create RolloutBuffer.
    
    Args:
        num_steps: Number of timesteps per rollout.
        num_envs: Number of parallel environments.
        obs_shape: Shape of observation tensor.
        device: PyTorch device.
        
    Returns:
        RolloutBuffer instance.
    """
    return RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_shape=obs_shape,
        device=device,
    )

