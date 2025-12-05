"""
Cognitive metrics for Coach Tal.

This module provides the mathematical primitives for computing:
    - Shannon entropy of policy distributions (opponent confusion)
    - User ease (negentropy / sharpness)
    - The full Cognitive Asymmetry objective J(s')
    - Soundness constraint checking
"""

from __future__ import annotations

import math
from typing import Dict

import chess


def entropy(policy: Dict[chess.Move, float], eps: float = 1e-10) -> float:
    """
    Compute Shannon entropy of a policy distribution over legal moves.
    
    H(π) = -Σ π(a) log π(a)
    
    Higher entropy means the policy is more "spread out" – the player has
    no clear best move and is more likely to make suboptimal choices.
    
    Args:
        policy: Dict mapping legal moves to probabilities (should sum to ~1).
        eps: Small constant to avoid log(0).
        
    Returns:
        Entropy in nats (natural log). Returns 0 if policy is empty.
    """
    if not policy:
        return 0.0
    
    h = 0.0
    for prob in policy.values():
        if prob > eps:
            h -= prob * math.log(prob)
    return h


def max_entropy(num_moves: int) -> float:
    """
    Compute maximum possible entropy for a uniform distribution over n moves.
    
    H_max = log(n)
    
    Args:
        num_moves: Number of legal moves.
        
    Returns:
        Maximum entropy in nats.
    """
    if num_moves <= 1:
        return 0.0
    return math.log(num_moves)


def user_ease(policy: Dict[chess.Move, float]) -> float:
    """
    Compute normalized user ease score.
    
    E_user(s) = 1 - H(π) / H_max
    
    A score of 1.0 means the user has a single clear best move (low entropy).
    A score of 0.0 means all moves look equally good (maximum entropy).
    
    Args:
        policy: Dict mapping legal moves to probabilities.
        
    Returns:
        User ease score in [0, 1].
    """
    if not policy:
        return 1.0  # No moves = trivially "easy" (game over)
    
    n = len(policy)
    h_max = max_entropy(n)
    
    if h_max == 0:
        return 1.0  # Only one move, maximum ease
    
    h = entropy(policy)
    return 1.0 - (h / h_max)


def cognitive_asymmetry_score(
    value: float,
    user_ease_score: float,
    opponent_entropy: float,
    lambda_psych: float = 0.3,
    gamma_confusion: float = 0.5,
) -> float:
    """
    Compute the Cognitive Asymmetry objective J(s').
    
    J(s') = V_φ(s') + λ * [E_user(s') + γ * H_opp(s')]
    
    This balances:
        - Objective soundness (value)
        - User comfort (high ease = low user entropy)
        - Opponent confusion (high opponent entropy)
    
    Args:
        value: Oracle evaluation V_φ(s') in [-1, 1].
        user_ease_score: Normalized user ease E_user in [0, 1].
        opponent_entropy: Raw opponent entropy H_opp in nats.
        lambda_psych: Weight for psychological factors vs objective truth.
        gamma_confusion: Weight for opponent confusion vs user ease.
        
    Returns:
        Combined cognitive asymmetry score.
    """
    # Normalize opponent entropy to roughly [0, 1] range for balanced weighting.
    # Typical entropy for ~30 legal moves with uniform dist is ~3.4 nats.
    # We use a soft normalization that doesn't clip but scales appropriately.
    normalized_opp_entropy = opponent_entropy / 3.5  # ~log(30)
    
    psychological_component = user_ease_score + gamma_confusion * normalized_opp_entropy
    
    return value + lambda_psych * psychological_component


def passes_soundness_constraint(
    value_after: float,
    value_before: float,
    delta: float = 0.15,
) -> bool:
    """
    Check if a move satisfies the soundness constraint.
    
    V_φ(s') >= V_φ(s) - δ
    
    The move must not worsen the position by more than δ (in value units).
    A typical δ of 0.15 corresponds to roughly ~0.3 pawns since value is in [-1, 1].
    
    Args:
        value_after: Oracle evaluation of position after the move.
        value_before: Oracle evaluation of position before the move.
        delta: Maximum allowed value drop.
        
    Returns:
        True if the move is sound, False otherwise.
    """
    return value_after >= value_before - delta


def compute_value_delta(value_after: float, value_before: float) -> float:
    """
    Compute the change in evaluation from a move.
    
    Positive means the position improved, negative means it worsened.
    
    Args:
        value_after: Evaluation after the move.
        value_before: Evaluation before the move.
        
    Returns:
        Delta V = V_after - V_before.
    """
    return value_after - value_before


def compute_entropy_delta(entropy_after: float, entropy_before: float) -> float:
    """
    Compute the change in entropy from a move.
    
    Positive means entropy increased (more confusion), negative means it decreased.
    
    Args:
        entropy_after: Entropy after the move.
        entropy_before: Entropy before the move.
        
    Returns:
        Delta H = H_after - H_before.
    """
    return entropy_after - entropy_before

