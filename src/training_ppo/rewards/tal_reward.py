"""
Tal Reward Engine for cognitive asymmetry training.

Computes the reward signal that encourages the agent to:
1. Win games (game outcome)
2. Create confusing positions (low survival mass)
3. Exploit opponent's value misjudgments (value gap)

The reward formula:
    R = R_outcome + α·(1 - M_surv) + β·Gap

Where:
    - R_outcome: +1 win, 0 draw, -1 loss
    - M_surv: Probability victim places on sound moves
    - Gap: Q_truth - V_victim (deception metric)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING

# Lazy JAX import to avoid import errors when JAX is not properly configured
if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

# Runtime JAX import (expected to be available in training/test environments)
from src.training_ppo.rewards.normalizer import RewardScaler

logger = logging.getLogger(__name__)


def _import_jax():
    """Lazy import JAX."""
    import jax
    import jax.numpy as jnp
    return jax, jnp


# Provide module-level handles for jax/jnp users
jax, jnp = _import_jax()


@dataclass
class TalRewardConfig:
    """Configuration for Tal reward computation."""
    alpha: float = 0.3        # Weight for survival mass penalty
    beta: float = 0.2         # Weight for value gap bonus
    delta_soundness: float = 0.15  # Threshold for sound move classification
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    
    # Minimum value gap to award bonus (avoid noise)
    min_gap_threshold: float = 0.05


class TalRewardEngine:
    """
    Computes cognitive asymmetry rewards for training.
    
    The goal is to train an agent that:
    1. Plays sound chess (wins games)
    2. Creates positions where the opponent is likely to blunder
    3. Exploits the gap between objective truth and opponent's perception
    
    Components:
        - Game Outcome: Standard win/loss/draw reward
        - Survival Mass: Probability victim places on sound moves
            - Low survival = opponent likely to blunder
        - Value Gap: Difference between MCTS value and victim's value
            - High gap = opponent underestimates danger
    
    Example:
        engine = TalRewardEngine(config)
        
        rewards = engine.compute_rewards(
            q_truth=mcts_values,
            v_victim=victim_values,
            pi_victim=victim_policy,
            game_outcomes=outcomes,
            sound_mask=legal_mask,  # or more refined soundness mask
        )
    """
    
    def __init__(self, config: Optional[TalRewardConfig] = None):
        """
        Initialize the reward engine.
        
        Args:
            config: TalRewardConfig with hyperparameters.
        """
        self.config = config or TalRewardConfig()
        
        # Reward normalization
        if self.config.normalize_rewards:
            self.scaler = RewardScaler(
                alpha=self.config.alpha,
                beta=self.config.beta,
                clip_range=self.config.reward_clip,
            )
        else:
            self.scaler = None
        
        logger.info(
            f"TalRewardEngine initialized: α={self.config.alpha}, "
            f"β={self.config.beta}, δ={self.config.delta_soundness}"
        )
    
    def compute_rewards(
        self,
        q_truth: jnp.ndarray,
        v_victim: jnp.ndarray,
        pi_victim: jnp.ndarray,
        game_outcomes: jnp.ndarray,
        sound_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Tal rewards for a batch of transitions.
        
        Args:
            q_truth: (B,) Agent's MCTS Q-values (ground truth).
            v_victim: (B,) Victim's raw network values (perception).
            pi_victim: (B, A) Victim's policy distribution.
            game_outcomes: (B,) Terminal rewards (+1/-1/0).
            sound_mask: (B, A) Boolean mask of sound moves.
            
        Returns:
            (B,) Composite Tal rewards.
        """
        # 1. Compute survival mass
        survival_mass = self._compute_survival_mass(pi_victim, sound_mask)
        
        # 2. Compute value gap (deception metric)
        value_gap = self._compute_value_gap(q_truth, v_victim)
        
        # 3. Combine components
        if self.scaler is not None:
            rewards = self.scaler.compute_and_normalize(
                game_outcomes,
                survival_mass,
                value_gap,
            )
        else:
            rewards = game_outcomes + \
                      self.config.alpha * (1 - survival_mass) + \
                      self.config.beta * value_gap
            
            # Clip if not normalizing
            if self.config.reward_clip is not None:
                rewards = jnp.clip(
                    rewards, 
                    -self.config.reward_clip, 
                    self.config.reward_clip,
                )
        
        return rewards
    
    def _compute_survival_mass(
        self,
        pi_victim: jnp.ndarray,
        sound_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute survival mass: probability victim places on sound moves.
        
        Low survival mass means the victim is likely to blunder,
        which is what we want the agent to create.
        
        Args:
            pi_victim: (B, A) Victim's policy.
            sound_mask: (B, A) Mask of sound (non-blundering) moves.
            
        Returns:
            (B,) Survival mass in [0, 1].
        """
        # Sum probability on sound moves
        survival_mass = jnp.sum(pi_victim * sound_mask.astype(jnp.float32), axis=-1)
        
        return survival_mass
    
    def _compute_value_gap(
        self,
        q_truth: jnp.ndarray,
        v_victim: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute value gap: difference between truth and perception.
        
        Positive gap means the victim underestimates the danger.
        We want to maximize this gap (create deceptive positions).
        
        Args:
            q_truth: (B,) MCTS-derived values (reality).
            v_victim: (B,) Victim's network values (perception).
            
        Returns:
            (B,) Value gap (positive = good for agent).
        """
        # Gap from victim's perspective (how much they're wrong)
        # If agent is better off than victim thinks, gap is positive
        gap = q_truth - v_victim
        
        # Optional: Only count significant gaps
        if self.config.min_gap_threshold > 0:
            gap = jnp.where(
                jnp.abs(gap) > self.config.min_gap_threshold,
                gap,
                jnp.zeros_like(gap),
            )
        
        return gap
    
    def compute_sound_mask(
        self,
        q_before: jnp.ndarray,
        q_after_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute which moves are "sound" (don't lose too much value).
        
        A move is sound if the value after the move is within
        delta_soundness of the value before.
        
        Args:
            q_before: (B,) Value before action.
            q_after_actions: (B, A) Value after each possible action.
            
        Returns:
            (B, A) Boolean mask of sound moves.
        """
        delta = self.config.delta_soundness
        
        # Move is sound if it doesn't drop value too much
        # q_after should be >= q_before - delta (accounting for perspective flip)
        sound_mask = q_after_actions >= (q_before[:, None] - delta)
        
        return sound_mask
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get reward component statistics for logging.
        
        Returns:
            Dictionary of metric name -> value.
        """
        if self.scaler is not None:
            return self.scaler.get_component_stats()
        return {}


class TalRewardEngineJIT:
    """
    JIT-compatible Tal reward computation.
    
    Pure functional implementation that can be used inside
    JIT-compiled training loops.
    """
    
    _jit_fn = None  # Cached JIT function
    
    @classmethod
    def compute_rewards(
        cls,
        q_truth,
        v_victim,
        pi_victim,
        game_outcomes,
        sound_mask,
        alpha: float = 0.3,
        beta: float = 0.2,
    ):
        """
        JIT-compiled reward computation.
        
        Args:
            q_truth: (B,) MCTS Q-values.
            v_victim: (B,) Victim values.
            pi_victim: (B, A) Victim policy.
            game_outcomes: (B,) Terminal rewards.
            sound_mask: (B, A) Sound move mask.
            alpha: Survival mass weight.
            beta: Value gap weight.
            
        Returns:
            Tuple of (rewards, metrics_dict).
        """
        if cls._jit_fn is None:
            @jax.jit
            def _compute(q_truth, v_victim, pi_victim, game_outcomes, sound_mask, alpha, beta):
                # Survival mass
                survival_mass = jnp.sum(pi_victim * sound_mask, axis=-1)
                
                # Value gap
                value_gap = q_truth - v_victim
                
                # Composite reward
                rewards = game_outcomes + alpha * (1 - survival_mass) + beta * value_gap
                
                # Metrics for logging
                metrics = {
                    "survival_mass_mean": jnp.mean(survival_mass),
                    "value_gap_mean": jnp.mean(value_gap),
                    "reward_mean": jnp.mean(rewards),
                    "reward_std": jnp.std(rewards),
                }
                
                return rewards, metrics
            
            cls._jit_fn = _compute
        
        return cls._jit_fn(q_truth, v_victim, pi_victim, game_outcomes, sound_mask, alpha, beta)


def create_reward_engine(config: Optional[Any] = None) -> TalRewardEngine:
    """
    Factory function to create TalRewardEngine.
    
    Args:
        config: Optional config (TalRewardConfig or nested config).
        
    Returns:
        TalRewardEngine instance.
    """
    if config is None:
        return TalRewardEngine()
    
    if hasattr(config, "reward"):
        config = config.reward
    
    if isinstance(config, TalRewardConfig):
        return TalRewardEngine(config)
    
    # Build from dict-like config
    reward_config = TalRewardConfig(
        alpha=getattr(config, "alpha", 0.3),
        beta=getattr(config, "beta", 0.2),
        delta_soundness=getattr(config, "delta_soundness", 0.15),
        normalize_rewards=getattr(config, "normalize_rewards", True),
        reward_clip=getattr(config, "reward_clip", 10.0),
    )
    
    return TalRewardEngine(reward_config)

