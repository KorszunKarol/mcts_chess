"""
Running statistics for reward normalization.

Maintains exponential moving average of mean and standard deviation
for normalizing rewards during training. This is critical for
training stability with the Tal reward.
"""

from __future__ import annotations

from typing import Tuple, Optional, TYPE_CHECKING
import logging

import numpy as np

# Lazy JAX import
if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _import_jax():
    """Lazy import JAX."""
    import jax
    import jax.numpy as jnp
    return jax, jnp


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    
    Uses Welford's online algorithm for numerically stable
    computation of running statistics.
    
    Example:
        normalizer = RunningMeanStd()
        
        for batch in data:
            normalizer.update(batch)
            normalized = normalizer.normalize(batch)
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_range: Optional[float] = 10.0,
    ):
        """
        Initialize running statistics.
        
        Args:
            epsilon: Small value for numerical stability.
            clip_range: Clip normalized values to [-clip, clip].
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self.clip_range = clip_range
    
    def update(self, x) -> None:
        """
        Update running statistics with new batch.
        
        Uses Welford's online algorithm for stability.
        
        Args:
            x: (B,) or (B, ...) batch of values.
        """
        # Flatten to 1D and convert to numpy
        x = np.asarray(x).flatten()
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: float,
        batch_var: float,
        batch_count: int,
    ) -> None:
        """
        Update from pre-computed moments.
        
        Args:
            batch_mean: Mean of new batch.
            batch_var: Variance of new batch.
            batch_count: Size of new batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / total_count
        
        # Update variance using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = max(new_var, self.epsilon)  # Ensure positive
        self.count = total_count
    
    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize values using running statistics.
        
        Args:
            x: Values to normalize.
            
        Returns:
            Normalized values with mean ~0 and std ~1.
        """
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        
        if self.clip_range is not None:
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return normalized
    
    def denormalize(self, x):
        """
        Reverse normalization.
        
        Args:
            x: Normalized values.
            
        Returns:
            Original scale values.
        """
        std = np.sqrt(self.var + self.epsilon)
        return x * std + self.mean
    
    @property
    def std(self) -> float:
        """Current standard deviation."""
        return float(np.sqrt(self.var + self.epsilon))
    
    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class RunningMeanStdJAX:
    """
    JAX-native running mean/std for use in JIT-compiled functions.
    
    Stores state as JAX arrays for compatibility with jax.jit.
    Note: Requires JAX to be properly installed.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        epsilon: float = 1e-8,
    ):
        """
        Initialize JAX-native running statistics.
        
        Args:
            shape: Shape of values to normalize (empty for scalar).
            epsilon: Small value for stability.
        """
        jax, jnp = _import_jax()
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = jnp.array(0.0)
        self.epsilon = epsilon
    
    def update(self, x) -> "RunningMeanStdJAX":
        """
        Update statistics (returns new instance for functional style).
        
        Args:
            x: Batch of values.
            
        Returns:
            New RunningMeanStdJAX with updated statistics.
        """
        jax, jnp = _import_jax()
        
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        new_rms = RunningMeanStdJAX(shape=self.mean.shape, epsilon=self.epsilon)
        new_rms.mean = new_mean
        new_rms.var = jnp.maximum(new_var, self.epsilon)
        new_rms.count = total_count
        
        return new_rms
    
    def normalize(self, x):
        """Normalize values."""
        jax, jnp = _import_jax()
        return (x - self.mean) / jnp.sqrt(self.var + self.epsilon)


class RewardScaler:
    """
    Reward scaling with multiple components.
    
    Tracks separate statistics for each reward component
    (game outcome, survival mass, value gap) for analysis.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.2,
        clip_range: float = 10.0,
    ):
        """
        Initialize reward scaler.
        
        Args:
            alpha: Weight for survival mass component.
            beta: Weight for value gap component.
            clip_range: Clip normalized rewards.
        """
        self.alpha = alpha
        self.beta = beta
        
        # Separate normalizers for each component
        self.outcome_stats = RunningMeanStd(clip_range=None)
        self.survival_stats = RunningMeanStd(clip_range=None)
        self.gap_stats = RunningMeanStd(clip_range=None)
        self.total_stats = RunningMeanStd(clip_range=clip_range)
    
    def compute_and_normalize(
        self,
        game_outcomes,
        survival_mass,
        value_gap,
    ):
        """
        Compute and normalize composite Tal reward.
        
        Args:
            game_outcomes: (B,) terminal rewards.
            survival_mass: (B,) victim survival mass [0, 1].
            value_gap: (B,) Q_truth - V_victim.
            
        Returns:
            (B,) normalized total rewards.
        """
        # Update individual component statistics
        self.outcome_stats.update(game_outcomes)
        self.survival_stats.update(survival_mass)
        self.gap_stats.update(value_gap)
        
        # Compute composite reward
        # Lower survival mass = better (more traps)
        # Higher value gap = better (more deception)
        total = game_outcomes + \
                self.alpha * (1 - survival_mass) + \
                self.beta * value_gap
        
        # Update and normalize total
        self.total_stats.update(total)
        normalized = self.total_stats.normalize(total)
        
        return normalized
    
    def get_component_stats(self) -> dict:
        """Get statistics for all components."""
        return {
            "outcome_mean": self.outcome_stats.mean,
            "outcome_std": self.outcome_stats.std,
            "survival_mean": self.survival_stats.mean,
            "survival_std": self.survival_stats.std,
            "gap_mean": self.gap_stats.mean,
            "gap_std": self.gap_stats.std,
            "total_mean": self.total_stats.mean,
            "total_std": self.total_stats.std,
        }

