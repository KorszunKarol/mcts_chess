"""
Style metrics for Tal personality verification.

These metrics verify that the agent is developing Tal-style play
(sacrifices, chaos, deception) rather than standard engine behavior.

Key Metrics:
    - Material Imbalance: Are we winning with sacrifices?
    - Chaos Index: Are we forcing "only moves"?
    - Agent Suicide Detection: Are we taking calculated risks or gambling?
"""

from __future__ import annotations

from typing import Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import jax.numpy as jnp


# Standard piece values (P, N, B, R, Q, K)
PIECE_VALUES = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])


def compute_material_imbalance(
    obs: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute material imbalance from observation tensor.
    
    Extracts piece counts from channels 0-11 and computes weighted
    material difference. Positive = Agent (White) ahead.
    
    Channel Layout:
        0-5:  Black pieces (bP, bN, bB, bR, bQ, bK)
        6-11: White pieces (wP, wN, wB, wR, wQ, wK)
    
    Args:
        obs: (B, C, H, W) observation tensor with C >= 12.
        device: Device for weight tensor.
        
    Returns:
        (B,) material imbalance per position.
        Positive = White/Agent ahead, Negative = sacrificing.
    
    Example:
        >>> obs = buffer.obs  # (128, 4096, 34, 8, 8)
        >>> material = compute_material_imbalance(obs.view(-1, 34, 8, 8))
        >>> # material > 0: ahead in material
        >>> # material < 0: sacrificing material (Tal-style!)
    """
    weights = PIECE_VALUES.to(obs.device)
    
    # Sum piece counts per type (B, 6) for each color
    # obs[:, 0:6] are black pieces, obs[:, 6:12] are white pieces
    black_counts = obs[:, 0:6, :, :].sum(dim=(2, 3))  # (B, 6)
    white_counts = obs[:, 6:12, :, :].sum(dim=(2, 3))  # (B, 6)
    
    # Weighted material sums
    black_material = (black_counts * weights).sum(dim=1)  # (B,)
    white_material = (white_counts * weights).sum(dim=1)  # (B,)
    
    # Agent plays White, so positive = agent ahead
    return white_material - black_material


def compute_chaos_index(
    legal_mask: torch.Tensor,
    sound_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute chaos index: number of sound moves available to opponent.
    
    A low chaos index means the opponent has few good options,
    which is the goal of Tal-style play (forcing "only moves").
    
    Args:
        legal_mask: (B, A) boolean mask of legal moves.
        sound_mask: (B, A) boolean mask of sound (non-blundering) moves.
        
    Returns:
        (B,) count of sound legal moves per position.
        Lower = more pressure on opponent.
    
    Example:
        >>> legal = env.get_legal_actions(state)
        >>> sound = reward_engine.compute_sound_mask(q_before, q_after)
        >>> chaos = compute_chaos_index(legal, sound)
        >>> # chaos < 3: opponent in trouble (only moves!)
        >>> # chaos > 10: calm position (not Tal-like)
    """
    # Intersection of legal and sound moves
    sound_legal = legal_mask.float() * sound_mask.float()
    
    # Count per position
    return sound_legal.sum(dim=-1)


def detect_agent_suicide(
    q_truth: torch.Tensor,
    threshold: float = -0.5,
) -> torch.Tensor:
    """
    Detect moves where agent played into a losing position.
    
    "Suicide" moves are those where the agent chose a position
    with Q < threshold just to chase deception rewards. These
    are bad - we want calculated risks, not gambling.
    
    Args:
        q_truth: (B,) MCTS Q-values for chosen moves.
        threshold: Q-value below which move is considered suicide.
            Default -0.5 means clearly losing position.
            
    Returns:
        (B,) binary mask where 1 = suicide move.
    
    Example:
        >>> suicide = detect_agent_suicide(q_truth, threshold=-0.5)
        >>> suicide_rate = suicide.mean()  # Target: < 0.05
    """
    return (q_truth < threshold).float()


def compute_blunder_induced(
    v_victim_before: torch.Tensor,
    v_victim_after: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Detect if we induced a blunder (victim's eval dropped significantly).
    
    When the victim's evaluation drops sharply after our move,
    it suggests we created a confusing position that led to
    a mistake. This is the "trap sprung" moment.
    
    Args:
        v_victim_before: (B,) Victim's value before agent's move.
        v_victim_after: (B,) Victim's value after agent's move.
        threshold: Minimum drop to count as blunder induced.
        
    Returns:
        (B,) binary mask where 1 = blunder induced.
        
    Note:
        This is an approximation. True blunder detection would
        require comparing victim's move to best move, but this
        captures the "confusion created" aspect.
    """
    # Victim's evaluation dropped = they made a mistake
    eval_drop = v_victim_before - v_victim_after
    return (eval_drop > threshold).float()


def compute_trap_success(
    value_gap: torch.Tensor,
    victim_blundered: torch.Tensor,
    gap_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Compute trap success: did high-gap positions lead to blunders?
    
    A "trap" is a position with high value gap (we know it's good,
    opponent thinks it's fine). Success = opponent actually blundered.
    
    Args:
        value_gap: (B,) Q_truth - V_victim per position.
        victim_blundered: (B,) binary mask of victim blunders.
        gap_threshold: Minimum gap to consider it a "trap".
        
    Returns:
        Tuple of:
            - trap_attempts: (B,) positions with high gap
            - trap_successes: (B,) high-gap positions where victim blundered
    """
    # Was this a trap attempt?
    trap_attempt = (value_gap > gap_threshold).float()
    
    # Did the trap work?
    trap_success = trap_attempt * victim_blundered
    
    return trap_attempt, trap_success


# ============================================================================
# JAX-compatible versions (for use in JIT-compiled training loops)
# ============================================================================

def compute_material_imbalance_jax(obs):
    """JAX version of material imbalance computation."""
    import jax.numpy as jnp
    
    weights = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    
    black_counts = obs[:, 0:6, :, :].sum(axis=(2, 3))
    white_counts = obs[:, 6:12, :, :].sum(axis=(2, 3))
    
    black_material = (black_counts * weights).sum(axis=1)
    white_material = (white_counts * weights).sum(axis=1)
    
    return white_material - black_material


def compute_chaos_index_jax(legal_mask, sound_mask):
    """JAX version of chaos index computation."""
    import jax.numpy as jnp
    
    sound_legal = legal_mask.astype(jnp.float32) * sound_mask.astype(jnp.float32)
    return sound_legal.sum(axis=-1)


def detect_agent_suicide_jax(q_truth, threshold: float = -0.5):
    """JAX version of suicide detection."""
    import jax.numpy as jnp
    
    return (q_truth < threshold).astype(jnp.float32)

