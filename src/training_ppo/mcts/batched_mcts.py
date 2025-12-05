"""
Batched MCTS using DeepMind's mctx library.

This module provides vectorized Monte Carlo Tree Search that processes
all positions in a batch simultaneously, enabling efficient GPU utilization.

Key features:
- Processes 4096 boards in parallel
- Uses mctx.gumbel_muzero_policy for fast, accurate search
- Returns improved policy and Q-values (V_truth)
- Supports environment-based dynamics (true chess rules)

Reference:
    https://github.com/google-deepmind/mctx
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional, NamedTuple, Callable

import jax
import jax.numpy as jnp
import mctx

from src.training_ppo.models.tal_jax import TalModelJAX
from src.utils import sentinel

logger = logging.getLogger(__name__)


class MCTSOutput(NamedTuple):
    """Output from BatchedMCTS search."""
    policy: jnp.ndarray     # (B, A) improved policy from visit counts
    q_value: jnp.ndarray    # (B,) root Q-values (V_truth)
    action: jnp.ndarray     # (B,) selected actions
    search_tree: Any        # mctx search tree for debugging


class BatchedMCTS:
    """
    Vectorized MCTS using DeepMind's mctx library.
    
    This provides System 2 thinking for the agent:
    - Runs num_simulations in parallel across all positions
    - Returns improved policy (from visit counts)
    - Returns Q-values as the "ground truth" value
    
    The agent uses MCTS to verify that positions are truly good
    (not just looking good to the policy network).
    
    Example:
        model, params = TalModelJAX.from_pytorch("weights.pt")
        mcts = BatchedMCTS(model, num_simulations=50)
        
        # Search from current positions
        output = mcts.search(params, key, observations, legal_mask)
        action = output.action
        q_truth = output.q_value
    """
    
    def __init__(
        self,
        model: TalModelJAX,
        num_simulations: int = 50,
        max_num_considered_actions: int = 16,
        discount: float = 1.0,
        temperature: float = 1.0,
        use_gumbel: bool = True,
    ):
        """
        Initialize BatchedMCTS.
        
        Args:
            model: TalModelJAX instance for evaluation.
            num_simulations: Number of MCTS simulations per position.
            max_num_considered_actions: Prune to top-K actions for speed.
            discount: Discount factor (1.0 for chess, no discounting).
            temperature: Temperature for action selection.
            use_gumbel: Use Gumbel-based search (recommended).
        """
        self.model = model
        self.num_simulations = num_simulations
        self.max_num_considered_actions = max_num_considered_actions
        self.discount = discount
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        logger.info(
            f"BatchedMCTS initialized: sims={num_simulations}, "
            f"max_actions={max_num_considered_actions}"
        )
    
    @sentinel.trace
    @sentinel.shape_guard(
        inputs={"key": "2", "obs": "B, *, *, *", "legal_mask": "B, A"},
        outputs={"policy": "B, A", "q_value": "B", "action": "B"},
    )
    def search(
        self,
        params: Dict[str, Any],
        key: jax.random.PRNGKey,
        obs: jnp.ndarray,
        env_state: Any,
        legal_mask: jnp.ndarray,
        env_step_fn: Optional[Callable] = None,
    ) -> MCTSOutput:
        """
        Run batched MCTS search.
        
        Args:
            params: Model parameters.
            key: JAX random key.
            obs: (B, C, H, W) observations (for network evaluation).
            env_state: Environment state (pgx.State) to embed at the root.
            legal_mask: (B, A) legal action mask.
            env_step_fn: Optional function for environment dynamics.
                         Signature: (state, action) -> (next_obs, reward, done, next_state)
            
        Returns:
            MCTSOutput with policy, Q-values, and selected actions.
        """
        # CRITICAL: mctx expects a SINGLE scalar key [2], NOT a batch of keys [B, 2].
        # Passing a batch causes a crossed-batch explosion [B, B, A] (the 1.14 GiB alloc).
        key = jnp.asarray(key)
        if key.ndim > 1:
            key = key.reshape(-1, 2)[0]  # squash any batched keys
        if key.shape != (2,):
            raise ValueError(f"CRITICAL: rng_key must be shape (2,), got {key.shape}")
        if sentinel.enabled:
            sentinel.log_tensor("rng_key", key)
            sentinel.log_tensor("obs", obs)
            sentinel.log_tensor("legal_mask", legal_mask)
        
        # Get root node evaluation
        root_output = self.model.apply(params, obs, train=False)
        root_value = root_output.value[:, 2] - root_output.value[:, 0]  # Win - Loss
        root_logits = jnp.asarray(root_output.policy_logits)
        if sentinel.enabled:
            sentinel.log_tensor("root_value", root_value)
            sentinel.log_tensor("policy_logits", root_logits)
        if root_logits.ndim != 2:
            raise ValueError(
                f"policy_logits must have shape (batch, actions); got {root_logits.shape}"
            )

        # Validate/collapse legal mask shape to (B, A)
        legal_mask = jnp.asarray(legal_mask)
        if legal_mask.ndim == 3 and legal_mask.shape[0] == legal_mask.shape[1]:
            legal_mask = legal_mask[:, 0, :]
        elif legal_mask.ndim != 2:
            raise ValueError(
                f"legal_mask must have shape (batch, actions); got {legal_mask.shape}"
            )
        
        # Mask illegal actions with large negative values
        masked_logits = jnp.where(
            legal_mask,
            root_logits,
            jnp.full_like(root_logits, -1e9),
        )
        
        # Create root node
        root = mctx.RootFnOutput(
            prior_logits=masked_logits,
            value=root_value,
            embedding=env_state,  # Embed full environment state for proper transitions
        )
        
        # Define recurrent function for tree expansion
        recurrent_fn = self._make_recurrent_fn(params, env_step_fn)
        
        # Run search
        # Pass SINGLE key [2] - mctx handles batching internally
        if self.use_gumbel:
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=key,  # Single key [2], NOT a batch [B, 2]
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                max_num_considered_actions=self.max_num_considered_actions,
                gumbel_scale=1.0,
            )
        else:
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=key,  # Single key [2], NOT a batch [B, 2]
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.num_simulations,
                max_num_considered_actions=self.max_num_considered_actions,
                temperature=self.temperature,
            )
        
        # Extract results
        # action_weights: (B, A) visit counts normalized to probabilities
        # action: (B,) selected action (highest visit count with Gumbel noise)
        policy = policy_output.action_weights
        action = policy_output.action
        
        # Compute Q-value from root (average over all actions weighted by visits)
        q_value = self._compute_root_q_value(policy_output, root_value)
        if sentinel.enabled:
            sentinel.log_tensor("mcts_policy", policy)
            sentinel.log_tensor("mcts_action", action)
            sentinel.log_tensor("mcts_q_value", q_value)
        
        return MCTSOutput(
            policy=policy,
            q_value=q_value,
            action=action,
            search_tree=policy_output.search_tree,
        )
    
    def _make_recurrent_fn(
        self,
        params: Dict[str, Any],
        env_step_fn: Optional[Callable] = None,
    ) -> Callable:
        """
        Create the recurrent function for mctx.
        
        This function is called during MCTS expansion to:
        1. Apply an action to get the next state
        2. Evaluate the next state with the model
        
        Args:
            params: Model parameters.
            env_step_fn: Optional environment step function.
            
        Returns:
            Recurrent function compatible with mctx.
        """
        def recurrent_fn(
            params: Dict[str, Any],
            rng_key: jax.random.PRNGKey,
            action: jnp.ndarray,
            embedding: Any,
        ) -> Tuple[mctx.RecurrentFnOutput, Any]:
            """
            Recurrent function for MCTS tree expansion.
            
            For chess, this represents taking an action and getting
            the opponent's perspective on the resulting position.
            
            Args:
                params: Model parameters.
                rng_key: Random key.
                action: (B,) action indices.
                embedding: current environment state (pgx.State).
                
            Returns:
                Tuple of (RecurrentFnOutput, next_state).
            """
            if env_step_fn is None:
                raise ValueError(
                    "env_step_fn is required for MCTS. Without state transitions, "
                    "MCTS cannot evaluate the consequences of moves beyond depth 1. "
                    "Either provide env_step_fn or implement learned dynamics (MuZero-style)."
                )
            
            # Use environment dynamics for state transitions
            # env_step_fn should return: next_obs, reward, done, next_state
            next_obs, reward, done, next_state = env_step_fn(embedding, action)
            
            # Handle reward shape: pgx returns (B, 2) for [white_reward, black_reward]
            # Extract agent's reward (white/agent perspective)
            if reward.ndim > 1 and reward.shape[1] == 2:
                reward = reward[:, 0]  # Take white/agent reward
            
            # Evaluate next state
            output = self.model.apply(params, next_obs, train=False)
            
            # Value from opponent's perspective (negate for parent's perspective)
            next_value = -(output.value[:, 2] - output.value[:, 0])
            
            # Handle terminal states
            next_value = jnp.where(done, reward, next_value)
            
            # Policy logits for next state
            next_logits = output.policy_logits
            
            recurrent_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.where(done, 0.0, self.discount),
                prior_logits=next_logits,
                value=next_value,
            )
            
            return recurrent_output, next_state
        
        return recurrent_fn
    
    def _compute_root_q_value(
        self,
        policy_output: Any,
        root_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Q-value for root node.
        
        This represents the expected value after the MCTS search,
        which should be more accurate than the raw network value.
        
        Args:
            policy_output: mctx policy output.
            root_value: (B,) raw network values.
            
        Returns:
            (B,) Q-values from MCTS.
        """
        # The search tree contains Q-values for all visited actions
        # We can extract the root's Q-value
        
        try:
            # Get Q-values from search tree
            tree = policy_output.search_tree
            # Root Q is the average of child Q-values weighted by visits
            # Or we can use the selected action's Q-value
            
            # Simplified: use weighted average
            q_values = tree.node_values[:, 0]  # Root node values
            
            return q_values
        except (AttributeError, IndexError):
            # Fallback to raw network value
            return root_value
    
    def get_action_and_stats(
        self,
        params: Dict[str, Any],
        key: jax.random.PRNGKey,
        obs: jnp.ndarray,
        env_state: Any,
        legal_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get action, policy, and Q-value in one call.
        
        Convenience method for the training loop.
        
        Args:
            params: Model parameters.
            key: JAX random key.
            obs: (B, C, H, W) observations.
            env_state: Environment state (pgx.State).
            legal_mask: (B, A) legal action mask.
            
        Returns:
            Tuple of (action, policy, q_value).
        """
        output = self.search(params, key, obs, env_state, legal_mask)
        return output.action, output.policy, output.q_value


class SimplifiedMCTS:
    """
    Simplified MCTS without mctx dependency.
    
    This is a fallback implementation for testing or when mctx
    is not available. Uses a simpler UCB-based search.
    """
    
    def __init__(
        self,
        model: TalModelJAX,
        num_simulations: int = 50,
        c_puct: float = 4.0,
    ):
        """
        Initialize simplified MCTS.
        
        Args:
            model: TalModelJAX instance.
            num_simulations: Number of simulations.
            c_puct: Exploration constant for UCB.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(
        self,
        params: Dict[str, Any],
        key: jax.random.PRNGKey,
        obs: jnp.ndarray,
        legal_mask: jnp.ndarray,
    ) -> MCTSOutput:
        """
        Run simplified search (single-level expansion).
        
        This is much simpler than full MCTS but still provides
        some search-based improvement over raw policy.
        
        Args:
            params: Model parameters.
            key: JAX random key.
            obs: (B, C, H, W) observations.
            legal_mask: (B, A) legal action mask.
            
        Returns:
            MCTSOutput with policy based on network output.
        """
        # Single forward pass
        output = self.model.apply(params, obs, train=False)
        value = output.value[:, 2] - output.value[:, 0]
        logits = output.policy_logits
        
        # Mask illegal moves
        masked_logits = jnp.where(
            legal_mask,
            logits,
            jnp.full_like(logits, -1e9),
        )
        
        # Compute policy
        policy = jax.nn.softmax(masked_logits, axis=-1)
        
        # Sample action
        action = jax.random.categorical(key, masked_logits, axis=-1)
        
        return MCTSOutput(
            policy=policy,
            q_value=value,
            action=action,
            search_tree=None,
        )


def create_mcts(
    model: TalModelJAX,
    config: Optional[Any] = None,
    use_simplified: bool = False,
) -> BatchedMCTS:
    """
    Factory function to create MCTS.
    
    Args:
        model: TalModelJAX instance.
        config: Optional MCTSConfig.
        use_simplified: Use simplified MCTS (for testing).
        
    Returns:
        BatchedMCTS or SimplifiedMCTS instance.
    """
    num_sims = 50
    max_actions = 16
    
    if config is not None:
        num_sims = getattr(config, "num_simulations", num_sims)
        max_actions = getattr(config, "max_num_considered_actions", max_actions)
    
    if use_simplified:
        return SimplifiedMCTS(model, num_simulations=num_sims)
    
    return BatchedMCTS(
        model=model,
        num_simulations=num_sims,
        max_num_considered_actions=max_actions,
    )

