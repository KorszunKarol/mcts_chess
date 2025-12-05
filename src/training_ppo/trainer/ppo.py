"""
PPO (Proximal Policy Optimization) Trainer.

Implements the PPO algorithm with:
- Clipped surrogate objective for policy
- Value function loss (MSE or clipped)
- Entropy bonus for exploration
- Gradient clipping

Reference:
    Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_ppo.storage.rollout_buffer import RolloutBuffer, Batch

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    lr: float = 3e-4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip range
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    target_kl: Optional[float] = None  # Early stopping KL threshold
    normalize_advantage: bool = True


class PPOTrainer:
    """
    PPO trainer for policy optimization.
    
    Implements the clipped surrogate objective:
        L^{CLIP} = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    
    Where:
        r_t = π_θ(a|s) / π_θ_old(a|s)
        A_t = Advantage estimate (from GAE)
        ε = clip_range (default 0.2)
    
    Example:
        model = TalModel()
        trainer = PPOTrainer(model, config)
        
        # After collecting rollout
        buffer.compute_returns_and_advantages(last_value)
        
        for epoch in range(ppo_epochs):
            metrics = trainer.update(buffer)
            if metrics["kl"] > target_kl:
                break
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[PPOConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model: Policy/value network (must have forward returning (value, policy_logits)).
            config: PPO hyperparameters.
            device: PyTorch device.
        """
        self.model = model
        self.config = config or PPOConfig()
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            eps=1e-5,  # Smaller epsilon for stability
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = None
        
        # Tracking
        self.update_count = 0
        
        logger.info(
            f"PPOTrainer initialized: lr={self.config.lr}, "
            f"clip={self.config.clip_range}, epochs={self.config.ppo_epochs}"
        )
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run PPO update on collected rollout.
        
        Args:
            buffer: RolloutBuffer with computed advantages.
            
        Returns:
            Dictionary of training metrics.
        """
        if not buffer.ready:
            raise RuntimeError("Buffer must have computed returns and advantages.")
        
        # Aggregate metrics across epochs
        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
        }
        
        num_batches = 0
        
        for epoch in range(self.config.ppo_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            
            for batch in buffer.get_batches(self.config.minibatch_size):
                metrics = self._update_batch(batch)
                
                # Accumulate metrics
                for key in total_metrics:
                    if key in metrics:
                        total_metrics[key] += metrics[key]
                
                epoch_kl += metrics.get("kl", 0.0)
                epoch_batches += 1
                num_batches += 1
            
            # Early stopping based on KL divergence
            avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if self.config.target_kl is not None and avg_epoch_kl > self.config.target_kl:
                logger.info(f"Early stopping at epoch {epoch + 1} due to KL={avg_epoch_kl:.4f}")
                break
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)
        
        total_metrics["epochs_run"] = epoch + 1
        total_metrics["batches_run"] = num_batches
        
        self.update_count += 1
        
        return total_metrics
    
    def _update_batch(self, batch: Batch) -> Dict[str, float]:
        """
        Update on a single minibatch.
        
        Args:
            batch: Minibatch of experience.
            
        Returns:
            Batch metrics.
        """
        # Forward pass
        value_preds, policy_logits = self._forward(batch.obs)
        
        # === Policy Loss ===
        # Compute log probs for taken actions (reuse for entropy)
        log_probs_all = F.log_softmax(policy_logits, dim=-1)
        log_probs = log_probs_all.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(log_probs - batch.old_log_probs)
        
        # Clipped surrogate objective
        policy_loss_1 = ratio * batch.advantages
        policy_loss_2 = torch.clamp(
            ratio, 
            1 - self.config.clip_range, 
            1 + self.config.clip_range,
        ) * batch.advantages
        
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # === Value Loss ===
        value_preds = value_preds.squeeze(-1) if value_preds.dim() > 1 else value_preds
        
        if self.config.clip_range_vf is not None:
            # Clipped value loss
            value_clipped = batch.values + torch.clamp(
                value_preds - batch.values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf,
            )
            value_loss_1 = F.mse_loss(value_preds, batch.returns, reduction="none")
            value_loss_2 = F.mse_loss(value_clipped, batch.returns, reduction="none")
            value_loss = torch.max(value_loss_1, value_loss_2).mean()
        else:
            value_loss = F.mse_loss(value_preds, batch.returns)
        
        # === Entropy Bonus ===
        probs = log_probs_all.exp()
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
        
        # === Total Loss ===
        total_loss = (
            policy_loss 
            + self.config.value_coef * value_loss 
            - self.config.entropy_coef * entropy
        )
        
        # === Backprop ===
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        
        self.optimizer.step()
        
        # === Compute Metrics ===
        with torch.no_grad():
            # Approximate KL divergence
            kl = (batch.old_log_probs - log_probs).mean()
            
            # Clip fraction (how often clipping was active)
            clip_fraction = (
                (ratio - 1).abs() > self.config.clip_range
            ).float().mean()
            
            # Explained variance
            explained_var = self._explained_variance(
                batch.returns,
                batch.values,
            )
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "kl": kl.item(),
            "clip_fraction": clip_fraction.item(),
            "explained_variance": explained_var,
        }
    
    def _forward(
        self, 
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            obs: (B, C, H, W) observations.
            
        Returns:
            Tuple of (value, policy_logits).
        """
        # Model outputs (value_probs, policy_logits)
        output = self.model(obs)
        
        if isinstance(output, tuple):
            value_probs, policy_logits = output
            # Convert W/D/L probs to scalar value
            value = value_probs[:, 2] - value_probs[:, 0]  # Win - Loss
        else:
            # Assume output has .value and .policy_logits
            value = output.value[:, 2] - output.value[:, 0]
            policy_logits = output.policy_logits
        
        return value, policy_logits
    
    def _get_log_probs(
        self,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probabilities for taken actions.
        
        Args:
            policy_logits: (B, A) raw logits.
            actions: (B,) action indices.
            
        Returns:
            (B,) log probabilities.
        """
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs
    
    def _compute_entropy(self, policy_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of policy.
        
        Args:
            policy_logits: (B, A) raw logits.
            
        Returns:
            Scalar entropy.
        """
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
    
    @staticmethod
    def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute explained variance (torch to avoid host transfer).
        """
        # flatten and detach to CPU minimally
        y_true_flat = y_true.view(-1).detach()
        y_pred_flat = y_pred.view(-1).detach()
        var_y = torch.var(y_true_flat)
        if var_y < 1e-10:
            return 0.0
        return float(1 - torch.var(y_true_flat - y_pred_flat) / var_y)
    
    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        logger.info(f"Loaded checkpoint from {path}")


def create_trainer(
    model: nn.Module,
    config: Optional[Any] = None,
    device: str = "cuda",
) -> PPOTrainer:
    """
    Factory function to create PPOTrainer.
    
    Args:
        model: Policy/value network.
        config: PPO config or nested config with .ppo attribute.
        device: PyTorch device.
        
    Returns:
        PPOTrainer instance.
    """
    if config is None:
        ppo_config = PPOConfig()
    elif hasattr(config, "ppo"):
        ppo_config = PPOConfig(
            lr=config.ppo.lr,
            clip_range=config.ppo.clip_range,
            clip_range_vf=config.ppo.clip_range_vf,
            entropy_coef=config.ppo.entropy_coef,
            value_coef=config.ppo.value_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            ppo_epochs=config.ppo.ppo_epochs,
            minibatch_size=config.ppo.minibatch_size,
        )
    elif isinstance(config, PPOConfig):
        ppo_config = config
    else:
        ppo_config = PPOConfig()
    
    return PPOTrainer(model, ppo_config, device)

