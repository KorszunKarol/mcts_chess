"""
Victim Model for cognitive asymmetry training.

The victim represents a bounded-rational player (System 1 thinking)
who uses raw policy outputs without MCTS. The elevated temperature
makes the policy "sloppier", simulating human-like mistakes.

Key properties:
- Frozen weights (no gradient updates)
- Elevated temperature (T=1.5) for softer policy
- Returns policy and value for reward computation
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from src.training_ppo.models.tal_jax import TalModelJAX, ModelOutput

logger = logging.getLogger(__name__)


class VictimOutput(NamedTuple):
    """Output from VictimModel."""
    value: jnp.ndarray         # (B,) scalar value in [-1, 1]
    policy: jnp.ndarray        # (B, A) policy probabilities
    entropy: jnp.ndarray       # (B,) policy entropy


class VictimModel:
    """
    Frozen victim model with elevated temperature.
    
    This simulates a bounded-rational opponent (System 1 thinking)
    who:
    - Uses raw policy without search (no MCTS)
    - Has elevated temperature (softer, more random policy)
    - Represents a ~1400 Elo player
    
    The victim's policy entropy is used to compute the "survival mass"
    component of the Tal reward: high entropy = more confusion.
    
    Example:
        model, params = TalModelJAX.from_pytorch("weights.pt")
        victim = VictimModel(model, params, temperature=1.5)
        
        # Get victim's response to position
        value, policy, entropy = victim(obs)
        action = sample(policy)
    """
    
    def __init__(
        self,
        model: TalModelJAX,
        params: Dict[str, Any],
        temperature: float = 1.5,
    ):
        """
        Initialize the victim model.
        
        Args:
            model: TalModelJAX instance.
            params: Frozen model parameters.
            temperature: Policy temperature (higher = more random).
        """
        self.model = model
        self.params = jax.tree.map(jax.lax.stop_gradient, params)  # Freeze
        self.temperature = temperature
        
        # JIT compile the forward pass
        self._forward = jax.jit(self._forward_fn)
        
        logger.info(f"VictimModel initialized with temperature={temperature}")
    
    def _forward_fn(self, obs: jnp.ndarray) -> VictimOutput:
        """
        Internal forward pass.
        
        Args:
            obs: (B, C, H, W) or (B, H, W, C) observation.
            
        Returns:
            VictimOutput with value, policy, and entropy.
        """
        # Forward through model
        output = self.model.apply(self.params, obs, train=False)
        
        # Apply temperature scaling to policy
        scaled_logits = output.policy_logits / self.temperature
        policy = jax.nn.softmax(scaled_logits, axis=-1)
        
        # Compute scalar value
        value = output.value[:, 2] - output.value[:, 0]  # Win - Loss
        
        # Compute entropy
        entropy = self._compute_entropy(policy)
        
        return VictimOutput(value=value, policy=policy, entropy=entropy)
    
    def __call__(self, obs: jnp.ndarray) -> VictimOutput:
        """
        Evaluate positions from victim's perspective.
        
        Args:
            obs: (B, C, H, W) or (B, H, W, C) observation batch.
            
        Returns:
            VictimOutput with:
                value: (B,) scalar values in [-1, 1]
                policy: (B, A) policy probabilities
                entropy: (B,) policy entropy
        """
        return self._forward(obs)
    
    def get_policy(
        self,
        obs: jnp.ndarray,
        legal_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Get policy distribution, optionally masked to legal moves.
        
        Args:
            obs: (B, C, H, W) observation batch.
            legal_mask: (B, A) boolean mask of legal actions.
            
        Returns:
            (B, A) policy probabilities (masked and renormalized if mask provided).
        """
        output = self._forward(obs)
        policy = output.policy
        
        if legal_mask is not None:
            # Mask illegal moves
            policy = jnp.where(legal_mask, policy, 0.0)
            # Renormalize
            policy = policy / (policy.sum(axis=-1, keepdims=True) + 1e-8)
        
        return policy
    
    def get_value(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get value estimate.
        
        Args:
            obs: (B, C, H, W) observation batch.
            
        Returns:
            (B,) scalar values in [-1, 1].
        """
        output = self._forward(obs)
        return output.value
    
    def get_entropy(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get policy entropy (measure of confusion).
        
        Args:
            obs: (B, C, H, W) observation batch.
            
        Returns:
            (B,) entropy values.
        """
        output = self._forward(obs)
        return output.entropy
    
    def sample_action(
        self,
        key: jax.random.PRNGKey,
        obs: jnp.ndarray,
        legal_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Sample actions from victim's policy.
        
        Args:
            key: JAX random key.
            obs: (B, C, H, W) observation batch.
            legal_mask: (B, A) boolean mask of legal actions.
            
        Returns:
            (B,) sampled action indices.
        """
        policy = self.get_policy(obs, legal_mask)
        
        # Sample from categorical distribution
        actions = jax.random.categorical(key, jnp.log(policy + 1e-8), axis=-1)
        
        return actions
    
    @staticmethod
    def _compute_entropy(policy: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Shannon entropy of policy.
        
        Args:
            policy: (B, A) probability distribution.
            
        Returns:
            (B,) entropy in nats.
        """
        # Avoid log(0)
        log_policy = jnp.log(policy + 1e-8)
        entropy = -jnp.sum(policy * log_policy, axis=-1)
        return entropy
    
    def compute_survival_mass(
        self,
        obs: jnp.ndarray,
        sound_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute survival mass: probability victim places on sound moves.
        
        This is a key metric for the Tal reward. Low survival mass means
        the victim is likely to blunder (good for agent).
        
        Args:
            obs: (B, C, H, W) observation batch.
            sound_mask: (B, A) mask of objectively sound moves.
            
        Returns:
            (B,) survival mass in [0, 1].
        """
        output = self._forward(obs)
        policy = output.policy
        
        # Sum probability mass on sound moves
        survival_mass = jnp.sum(policy * sound_mask, axis=-1)
        
        return survival_mass


def create_victim(
    model: TalModelJAX,
    params: Dict[str, Any],
    config: Optional[Any] = None,
) -> VictimModel:
    """
    Factory function to create VictimModel.
    
    Args:
        model: TalModelJAX instance.
        params: Model parameters.
        config: Optional VictimConfig.
        
    Returns:
        VictimModel instance.
    """
    temperature = 1.5
    if config is not None:
        temperature = getattr(config, "temperature", temperature)
    
    return VictimModel(model, params, temperature=temperature)

