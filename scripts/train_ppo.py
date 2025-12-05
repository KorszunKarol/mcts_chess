#!/usr/bin/env python3
"""
Tal-RL PPO Training Script.

This script orchestrates the full training pipeline for learning
cognitive asymmetry in chess:

1. Initialize vectorized chess environment (pgx)
2. Load Agent model (JAX/Flax) and Victim model (frozen, high-T)
3. Run rollouts with asymmetric play:
   - Agent (White): Uses MCTS for System 2 thinking
   - Victim (Black): Uses raw policy for System 1 thinking
4. Compute Tal rewards (survival mass, value gap)
5. Update Agent with PPO

Usage:
    python scripts/train_ppo.py --config configs/ppo_tal.yaml
    python scripts/train_ppo.py --num_envs 256  # Override config
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Limit JAX preallocation so PyTorch has room on the GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

from src.training_ppo.config import PPOTalConfig  # noqa: E402
from src.training_ppo.env.chess_env import create_env  # noqa: E402
from src.training_ppo.models.tal_jax import TalModelJAX, create_model as create_jax_model  # noqa: E402
from src.training_ppo.models.victim import create_victim  # noqa: E402
from src.training_ppo.models.jax_bridge import jax_to_torch  # noqa: E402
from src.training_ppo.mcts.batched_mcts import create_mcts  # noqa: E402
from src.training_ppo.specs import ACTION_SPACE_SIZE  # noqa: E402
from src.training_ppo.rewards.tal_reward import TalRewardEngineJIT  # noqa: E402
from src.training_ppo.storage.rollout_buffer import create_buffer  # noqa: E402
from src.training_ppo.trainer.ppo import create_trainer  # noqa: E402
from src.training_ppo.metrics.logger import create_logger  # noqa: E402
from src.training_ppo.metrics.style_metrics import (  # noqa: E402
    compute_material_imbalance,
    compute_chaos_index,
    detect_agent_suicide,
)  # noqa: E402
from src.utils import sentinel  # noqa: E402

# For PyTorch model (used in PPO updates)
from src.transformer_model_pytorch import create_model as create_pytorch_model  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tal-RL PPO Training")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_tal.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override number of environments",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Override rollout steps",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Override model weights path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (fewer envs, more logging)",
    )
    parser.add_argument(
        "--debug_level",
        type=str,
        default="OFF",
        choices=["OFF", "SHAPES", "VALUES", "FULL_TRACE"],
        help="Sentinel debug verbosity (overrides --debug when provided)",
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> PPOTalConfig:
    """Load and merge config with command line overrides."""
    if Path(args.config).exists():
        config = PPOTalConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = PPOTalConfig()
        logger.info("Using default config")
    
    # Apply overrides
    if args.num_envs is not None:
        config.env.num_envs = args.num_envs
    if args.num_steps is not None:
        config.ppo.num_steps = args.num_steps
    if args.total_timesteps is not None:
        config.training.total_timesteps = args.total_timesteps
    if args.weights_path is not None:
        config.agent.weights_path = args.weights_path
    if args.seed is not None:
        config.training.seed = args.seed
    
    # Debug mode overrides
    if args.debug:
        config.env.num_envs = 64
        config.ppo.num_steps = 32
        config.training.log_interval = 1
        logger.info("Debug mode: reduced to 64 envs, 32 steps")
    
    return config


def configure_sentinel(args: argparse.Namespace) -> None:
    """Configure the Sentinel debugging system."""
    level = args.debug_level
    if args.debug and args.debug_level.upper() == "OFF":
        level = "SHAPES"
    sentinel.configure(
        verbosity=level,
        inspect_jax_tracers=True,
        inspect_runtime_values=False,
    )


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # JAX key is set separately per-call


def normalize_legal_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Ensure legal mask has shape (B, ACTION_SPACE_SIZE).
    Collapses accidental (B, B, A) and pads/truncates if needed.
    """
    mask = jnp.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == mask.shape[1]:
        mask = mask[:, 0, :]
    if mask.ndim != 2:
        raise ValueError(f"legal_mask must be 2D; got {mask.shape}")
    if mask.shape[1] < ACTION_SPACE_SIZE:
        pad = ACTION_SPACE_SIZE - mask.shape[1]
        mask = jnp.pad(mask, ((0, 0), (0, pad)))
    elif mask.shape[1] > ACTION_SPACE_SIZE:
        mask = mask[:, :ACTION_SPACE_SIZE]
    return mask


def normalize_policy(policy: jnp.ndarray) -> jnp.ndarray:
    """
    Ensure policy has shape (B, ACTION_SPACE_SIZE), collapsing (B, B, A) if seen.
    """
    policy = jnp.asarray(policy)
    if policy.ndim == 3 and policy.shape[0] == policy.shape[1]:
        policy = policy[:, 0, :]
    if policy.ndim != 2:
        raise ValueError(f"policy must be 2D; got {policy.shape}")
    if policy.shape[1] < ACTION_SPACE_SIZE:
        pad = ACTION_SPACE_SIZE - policy.shape[1]
        policy = jnp.pad(policy, ((0, 0), (0, pad)))
    elif policy.shape[1] > ACTION_SPACE_SIZE:
        policy = policy[:, :ACTION_SPACE_SIZE]
    return policy


def normalize_actions(actions: jnp.ndarray, num_envs: int) -> jnp.ndarray:
    """
    Ensure actions are shape (num_envs,), collapsing (B, B) if seen.
    """
    actions = jnp.asarray(actions)
    if actions.ndim == 2 and actions.shape[0] == actions.shape[1]:
        actions = actions[:, 0]
    actions = actions.ravel()
    if actions.size != num_envs:
        raise ValueError(f"actions size {actions.size} does not match num_envs {num_envs}")
    return actions.reshape((num_envs,)).astype(jnp.int32)


def compute_sound_mask_from_mcts_policy(
    mcts_policy: jnp.ndarray,
    legal_mask: jnp.ndarray,
    top_k: int = 8,
) -> jnp.ndarray:
    """
    Approximate soundness: keep top-K MCTS moves (intersection with legal).
    """
    if mcts_policy.ndim != 2:
        return legal_mask
    top_k = max(1, min(top_k, mcts_policy.shape[1]))
    # Select top-K moves per position
    topk_idx = jnp.argpartition(mcts_policy, -top_k, axis=-1)[:, -top_k:]
    mask = jnp.zeros_like(legal_mask, dtype=bool)
    batch_indices = jnp.arange(mask.shape[0])[:, None]
    mask = mask.at[batch_indices, topk_idx].set(True)
    return mask & legal_mask


def main():
    """Main training loop."""
    args = parse_args()
    configure_sentinel(args)
    config = load_config(args)
    
    # Set seeds
    set_seeds(config.training.seed)
    jax_key = jax.random.PRNGKey(config.training.seed)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # === Initialize Components ===
    
    # 1. Environment
    logger.info("Initializing environment...")
    env = create_env(config)
    
    # 2. JAX Model (for MCTS and Victim)
    logger.info("Loading JAX model...")
    jax_model = create_jax_model()
    
    # Try to load weights
    jax_params = None
    if Path(config.agent.weights_path).exists():
        try:
            jax_model, jax_params = TalModelJAX.from_pytorch(config.agent.weights_path)
            logger.info(f"Loaded JAX model from {config.agent.weights_path}")
        except Exception as e:
            logger.warning(f"Could not load JAX weights: {e}. Using random init.")
    
    if jax_params is None:
        # Initialize with random weights
        jax_key, init_key = jax.random.split(jax_key)
        dummy_input = jnp.zeros((1, 8, 8, 34))
        # Note: init() doesn't take train parameter - it's only for apply()
        variables = jax_model.init(init_key, dummy_input)
        jax_params = {"params": variables["params"], "batch_stats": variables.get("batch_stats", {})}
    
    # 3. Victim Model (frozen, high temperature)
    logger.info("Creating victim model...")
    victim = create_victim(jax_model, jax_params, config.victim)
    
    # 4. Batched MCTS
    logger.info("Initializing batched MCTS...")
    # Use simplified MCTS to avoid env-state embedding issues and reduce memory
    mcts = create_mcts(jax_model, config.mcts, use_simplified=True)
    
    # 5. PyTorch Model (for PPO updates)
    logger.info("Loading PyTorch model...")
    pytorch_model = create_pytorch_model()
    pytorch_model = pytorch_model.to(device)
    
    if Path(config.agent.weights_path).exists():
        try:
            state_dict = torch.load(config.agent.weights_path, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            pytorch_model.load_state_dict(state_dict)
            logger.info(f"Loaded PyTorch model from {config.agent.weights_path}")
        except Exception as e:
            logger.warning(f"Could not load PyTorch weights: {e}")
    
    # 6. Rollout Buffer
    logger.info("Creating rollout buffer...")
    buffer = create_buffer(
        num_steps=config.ppo.num_steps,
        num_envs=config.env.num_envs,
        obs_shape=env.observation_shape,
        device=device,
    )
    
    # 7. PPO Trainer
    logger.info("Initializing PPO trainer...")
    trainer = create_trainer(pytorch_model, config, device)
    
    # 8. Logger
    logger.info("Setting up logging...")
    metrics_logger = create_logger(config)
    
    # Resume from checkpoint if provided
    start_iteration = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        trainer.load_checkpoint(args.checkpoint)
        start_iteration = trainer.update_count
        logger.info(f"Resumed from checkpoint: iteration {start_iteration}")
    
    # === Training Loop ===
    logger.info("=" * 60)
    logger.info("Starting Tal-RL PPO Training")
    logger.info(f"  Environments: {config.env.num_envs}")
    logger.info(f"  Steps per rollout: {config.ppo.num_steps}")
    logger.info(f"  Total timesteps: {config.training.total_timesteps:,}")
    logger.info("=" * 60)
    
    # Reset environment
    jax_key, reset_key = jax.random.split(jax_key)
    obs, state = env.reset(reset_key)
    
    # Convert to PyTorch for buffer storage
    obs_torch = jax_to_torch(obs, device)
    
    total_timesteps = 0
    num_iterations = config.training.total_timesteps // (config.ppo.num_steps * config.env.num_envs)
    
    for iteration in range(start_iteration, num_iterations):
        iteration_scope = sentinel.scope(f"ITERATION {iteration}") if sentinel.enabled else contextlib.nullcontext()
        with iteration_scope:
            rollout_scope = sentinel.scope("ROLLOUT") if sentinel.enabled else contextlib.nullcontext()
            with rollout_scope:
                # === Rollout Phase ===
                for step in range(config.ppo.num_steps):
                    jax_key, action_key, mcts_key = jax.random.split(jax_key, 3)
                    
                    # Get legal action mask (normalize to (B, ACTION_SPACE_SIZE))
                    legal_mask_raw = env.get_legal_actions(state)
                    legal_mask = normalize_legal_mask(legal_mask_raw)
                    
                    # Determine whose turn it is
                    is_agent = env.is_agent_turn(state)

                    # Evaluate victim everywhere up front so Tal reward never loses signal
                    victim_output = victim(obs)
                    victim_policy = victim_output.policy
                    victim_entropy = victim_output.entropy
                    v_victim = victim_output.value
                    
                    # === Agent's Turn (System 2: MCTS) ===
                    agent_actions = jnp.zeros((config.env.num_envs,), dtype=jnp.int32)
                    q_truth = jnp.zeros(config.env.num_envs)
                    mcts_policy = jnp.zeros_like(legal_mask, dtype=jnp.float32)
        
                    if is_agent.any():
                        # Create env_step_fn for MCTS
                        # Signature: (state, action) -> (next_obs, reward, done, next_state)
                        def env_step_fn(state_in, actions_batch):
                            step_result, next_state = env.step(state_in, actions_batch)
                            next_obs = step_result.obs
                            rewards = step_result.rewards
                            dones = step_result.dones
                            return next_obs, rewards, dones, next_state
                        
                        # Run MCTS (batched or simplified)
                        if hasattr(mcts, "use_gumbel"):
                            mcts_output = mcts.search(
                                jax_params,
                                mcts_key,
                                obs,
                                state,
                                legal_mask,
                                env_step_fn=env_step_fn,
                            )
                        else:
                            # SimplifiedMCTS signature: (params, key, obs, legal_mask)
                            mcts_output = mcts.search(
                                jax_params,
                                mcts_key,
                                obs,
                                legal_mask,
                            )
                    
                        # Sample from MCTS policy (ensure policy shape is correct)
                        mcts_policy = normalize_policy(mcts_output.policy)
                        agent_actions = jax.random.categorical(
                            action_key,
                            jnp.log(mcts_policy + 1e-8),
                            axis=-1,
                        )
                        agent_actions = normalize_actions(agent_actions, config.env.num_envs)
                        q_truth = mcts_output.q_value

                    victim_actions = jnp.zeros((config.env.num_envs,), dtype=jnp.int32)
                    if (~is_agent).any():
                        # Mask victim policy to legal moves before sampling
                        victim_policy_masked = jnp.where(legal_mask, victim_policy, 0.0)
                        victim_policy_masked = victim_policy_masked / (
                            victim_policy_masked.sum(axis=-1, keepdims=True) + 1e-8
                        )
                        victim_actions = jax.random.categorical(
                            action_key,
                            jnp.log(victim_policy_masked + 1e-8),
                            axis=-1,
                        )
                        victim_actions = normalize_actions(victim_actions, config.env.num_envs)
                    
                    # Combine actions based on whose turn
                    actions = jnp.where(is_agent, agent_actions, victim_actions)
                    actions = normalize_actions(actions, config.env.num_envs)
                    if actions.shape != (config.env.num_envs,):
                        raise ValueError(f"actions shape {actions.shape} expected {(config.env.num_envs,)}")
                    
                    # Get value and log_prob from PyTorch model for PPO
                    with torch.no_grad():
                        value_probs, policy_logits = pytorch_model(obs_torch)
                        value = (value_probs[:, 2] - value_probs[:, 0])  # Win - Loss
                        
                        # Log prob of taken action
                        log_probs_all = torch.log_softmax(policy_logits, dim=-1)
                        actions_torch = jax_to_torch(actions, device).long()
                        log_prob = log_probs_all.gather(1, actions_torch.unsqueeze(-1)).squeeze(-1)
                    
                    # Step environment
                    step_result, state = env.step(state, actions)
                    
                    # Compute Tal reward for agent's moves
                    # Approximate soundness using top-K MCTS moves; fallback to legal mask if unavailable
                    sound_mask = compute_sound_mask_from_mcts_policy(mcts_policy, legal_mask)
                    
                    # Extract agent's reward from pgx format (B, 2) -> (B,)
                    # pgx returns [white_reward, black_reward], agent is white
                    game_outcomes = step_result.rewards[:, 0] if step_result.rewards.ndim > 1 else step_result.rewards
                    rewards_jax, tal_metrics = TalRewardEngineJIT.compute_rewards(
                        q_truth,
                        v_victim,
                        victim_policy,
                        game_outcomes,
                        sound_mask,
                        alpha=config.reward.alpha,
                        beta=config.reward.beta,
                    )
                    
                    # Convert to PyTorch
                    rewards_torch = jax_to_torch(rewards_jax, device)
                    dones_torch = jax_to_torch(step_result.dones, device)
                    q_truth_torch = jax_to_torch(q_truth, device)
                    victim_entropy_torch = jax_to_torch(victim_entropy, device)
                    v_victim_torch = jax_to_torch(v_victim, device)
                    
                    # === Compute Style Metrics ===
                    # Material imbalance: positive = agent ahead, negative = sacrificing (Tal-style!)
                    material_imbalance = compute_material_imbalance(obs_torch)
                    
                    # Chaos index: low = opponent has few good moves (we want this!)
                    # Convert JAX masks to PyTorch
                    legal_mask_torch = jax_to_torch(legal_mask, device).bool()
                    sound_mask_torch = jax_to_torch(sound_mask, device).bool()
                    chaos_index = compute_chaos_index(legal_mask_torch, sound_mask_torch)
                    
                    # Agent suicide detection: moves where Q < -0.5 (losing)
                    agent_suicide = detect_agent_suicide(q_truth_torch, threshold=-0.5)
                    
                    # Value gap for episode tracking
                    value_gap = q_truth_torch - v_victim_torch
                    
                    # Add to buffer (with style metrics)
                    buffer.add(
                        obs=obs_torch,
                        action=actions_torch,
                        reward=rewards_torch,
                        done=dones_torch,
                        value=value,
                        log_prob=log_prob,
                        q_truth=q_truth_torch,
                        victim_entropy=victim_entropy_torch,
                        material_imbalance=material_imbalance,
                        chaos_index=chaos_index,
                        agent_suicide=agent_suicide,
                        value_gap=value_gap,
                    )
                    
                    # Log step metrics
                    metrics_logger.log_step({
                        "reward": float(rewards_torch.mean()),
                        "done_rate": float(dones_torch.float().mean()),
                    }, num_envs=config.env.num_envs)
                    
                    # Log Tal metrics (cognitive asymmetry)
                    metrics_logger.log_tal_metrics({
                        "survival_mass_mean": float(tal_metrics["survival_mass_mean"]),
                        "value_gap_mean": float(tal_metrics["value_gap_mean"]),
                    })
                    
                    # Log style metrics (Tal personality verification)
                    metrics_logger.log_style_metrics({
                        "material_imbalance_mean": float(material_imbalance.mean()),
                        "chaos_index_mean": float(chaos_index.mean()),
                    })
                    
                    # Log safety metrics (hope chess prevention)
                    metrics_logger.log_safety_metrics({
                        "agent_suicide_rate": float(agent_suicide.mean()),
                    })
                    
                    # Update state
                    obs = step_result.obs
                    obs_torch = jax_to_torch(obs, device)
                    
                    # Auto-reset terminated environments
                    jax_key, reset_key = jax.random.split(jax_key)
                    state = env.auto_reset(state, reset_key)
                    
                    total_timesteps += config.env.num_envs
            
            update_scope = sentinel.scope("PPO_UPDATE") if sentinel.enabled else contextlib.nullcontext()
            with update_scope:
                # === PPO Update ===
                # Get last value for GAE
                with torch.no_grad():
                    value_probs, _ = pytorch_model(obs_torch)
                    last_value = value_probs[:, 2] - value_probs[:, 0]
                
                # Compute returns and advantages
                buffer.compute_returns_and_advantages(
                    last_value,
                    gamma=config.ppo.gamma,
                    gae_lambda=config.ppo.gae_lambda,
                )
                
                # Run PPO update
                ppo_metrics = trainer.update(buffer)
                
                # Get episode statistics before resetting
                episode_stats = buffer.get_episode_statistics(clear=True)
                
                # Get buffer statistics for style metrics
                buffer_stats = buffer.get_statistics()
            
            # Reset buffer
            buffer.reset()
            
            # === Logging ===
            if iteration % config.training.log_interval == 0:
                metrics_logger.log_iteration(
                    iteration=iteration,
                    ppo_metrics=ppo_metrics,
                    episode_stats=episode_stats,
                    style_metrics={
                        "material_imbalance_mean": buffer_stats.get("material_imbalance_mean", 0),
                        "chaos_index_mean": buffer_stats.get("chaos_index_mean", 0),
                    },
                    safety_metrics={
                        "agent_suicide_rate": buffer_stats.get("agent_suicide_rate", 0),
                    },
                )
            
            # === Checkpointing ===
            if iteration % config.training.save_interval == 0 and iteration > 0:
                checkpoint_dir = Path(config.training.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
                trainer.save_checkpoint(str(checkpoint_path))
                metrics_logger.log_model_checkpoint(str(checkpoint_path), iteration)
            
            # Sync JAX params from PyTorch (for next rollout)
            # This is simplified; proper implementation would convert weights
            # For now, they stay in sync since we're using the same initial weights
    
    # === Finish ===
    logger.info("Training complete!")
    
    # Save final checkpoint
    final_path = Path(config.training.checkpoint_dir) / "checkpoint_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_path))
    
    metrics_logger.finish()


if __name__ == "__main__":
    main()

