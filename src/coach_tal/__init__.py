"""
Coach Tal: Cognitive Asymmetry Chess Engine

This module implements "Coach Tal" – a decision-support agent that optimizes
for Cognitive Asymmetry rather than raw win probability. The system selects
moves that maximize the gap between the user's cognitive ease and the
opponent's cognitive burden, subject to soundness constraints.

Key Components:
    - metrics: Entropy, user-ease, and J(s') scoring functions
    - evaluator: Lightweight transformer inference wrapper
    - agents: OpponentModel and UserModel proxies
    - explainer: Natural language explanations for move choices
    - selector: Root-level move re-ranking using cognitive asymmetry

Example Usage:
    from src.coach_tal import CoachTalSelector, CoachTalConfig, Explainer
    
    config = CoachTalConfig(weights_path="path/to/model.keras")
    selector = CoachTalSelector(config)
    explainer = Explainer()
    
    board = chess.Board()
    result = selector.select_from_board(board)
    analysis = explainer.explain(result, board)
    print(f"Play {analysis.move_san}: {analysis.primary_reason}")
"""

# Metrics (pure functions, no heavy dependencies)
from src.coach_tal.metrics import (
    entropy,
    user_ease,
    cognitive_asymmetry_score,
    passes_soundness_constraint,
    max_entropy,
    compute_value_delta,
    compute_entropy_delta,
)

# Evaluator (lazy-loads TensorFlow/PyTorch on first use)
from src.coach_tal.evaluator import TransformerEvaluator

# Agent proxies
from src.coach_tal.agents import OpponentModel, UserModel, create_agent_pair

# Explainer
from src.coach_tal.explainer import Explainer, MoveAnalysis

# Selector (main entry point)
from src.coach_tal.selector import (
    CoachTalSelector,
    CoachTalConfig,
    MoveCandidate,
    SelectionResult,
)

__all__ = [
    # Metrics
    "entropy",
    "user_ease",
    "cognitive_asymmetry_score",
    "passes_soundness_constraint",
    "max_entropy",
    "compute_value_delta",
    "compute_entropy_delta",
    # Evaluator
    "TransformerEvaluator",
    # Agents
    "OpponentModel",
    "UserModel",
    "create_agent_pair",
    # Explainer
    "Explainer",
    "MoveAnalysis",
    # Selector
    "CoachTalSelector",
    "CoachTalConfig",
    "MoveCandidate",
    "SelectionResult",
]

