"""
Coach Tal move selector with cognitive asymmetry re-ranking.

This module provides the core move selection logic that:
    1. Takes candidate moves from MCTS (or raw policy)
    2. Evaluates each candidate using cognitive asymmetry metrics
    3. Re-ranks and selects the move that maximizes J(s') subject to soundness
    4. Returns detailed analysis for explanation generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess

from src.coach_tal.evaluator import TransformerEvaluator
from src.coach_tal.agents import OpponentModel, UserModel, create_agent_pair
from src.coach_tal.metrics import (
    cognitive_asymmetry_score,
    passes_soundness_constraint,
    entropy,
    user_ease,
    compute_value_delta,
    compute_entropy_delta,
)

logger = logging.getLogger(__name__)


@dataclass
class CoachTalConfig:
    """
    Configuration for Coach Tal move selection.
    
    Attributes:
        weights_path: Path to transformer model weights.
        use_pytorch: Whether to use PyTorch backend (vs Keras).
        lambda_psych: Weight for psychological factors vs objective value.
        gamma_confusion: Weight for opponent confusion vs user ease.
        delta_soundness: Maximum allowed value drop for soundness constraint.
        top_k_candidates: Number of top MCTS moves to consider for re-ranking.
        opponent_temperature: Softmax temperature for opponent model.
        user_temperature: Softmax temperature for user model.
        enabled: Whether Coach Tal re-ranking is active (for A/B testing).
    """
    
    weights_path: str = ""
    use_pytorch: bool = False
    lambda_psych: float = 0.3
    gamma_confusion: float = 0.5
    delta_soundness: float = 0.15
    top_k_candidates: int = 5
    opponent_temperature: float = 1.2
    user_temperature: float = 1.0
    enabled: bool = True


@dataclass
class MoveCandidate:
    """
    Analysis of a single candidate move.
    
    Stores all metrics needed for ranking and explanation.
    """
    
    move: chess.Move
    mcts_score: float  # Original MCTS visit proportion or policy prob
    value_after: float  # V_φ(s') after the move
    opponent_entropy: float  # H_opp in the resulting position
    user_ease: float  # E_user in the resulting position (after opp reply)
    j_score: float  # Combined cognitive asymmetry score
    is_sound: bool  # Passes soundness constraint
    
    # Deltas relative to best objective move
    value_delta: float = 0.0
    entropy_delta: float = 0.0


@dataclass
class SelectionResult:
    """
    Result of Coach Tal move selection.
    
    Contains the chosen move and analysis of all candidates.
    """
    
    chosen_move: chess.Move
    chosen_analysis: MoveCandidate
    all_candidates: List[MoveCandidate]
    root_value: float  # V_φ(s) of the starting position
    fallback_used: bool  # True if we fell back to MCTS best due to no sound moves


class CoachTalSelector:
    """
    Selects moves using cognitive asymmetry optimization.
    
    This is the main entry point for Coach Tal. Given a position and
    candidate moves (from MCTS or raw policy), it:
        1. Evaluates each candidate's resulting position
        2. Computes cognitive metrics (opponent entropy, user ease)
        3. Ranks by J(s') subject to soundness constraint
        4. Returns the best move with full analysis
    
    Example:
        config = CoachTalConfig(weights_path="model.keras")
        selector = CoachTalSelector(config)
        
        # After MCTS search gives you candidate moves:
        result = selector.select(board, mcts_policy)
        print(f"Play {result.chosen_move} - J={result.chosen_analysis.j_score:.3f}")
    """
    
    def __init__(self, config: CoachTalConfig) -> None:
        """
        Initialize the selector.
        
        Args:
            config: Configuration for Coach Tal.
        """
        self.config = config
        
        # Lazy initialization of models
        self._evaluator: Optional[TransformerEvaluator] = None
        self._opponent: Optional[OpponentModel] = None
        self._user: Optional[UserModel] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Initialize models on first use."""
        if self._initialized:
            return
        
        self._evaluator = TransformerEvaluator(
            weights_path=self.config.weights_path,
            use_pytorch=self.config.use_pytorch,
            temperature=1.0,
        )
        
        self._opponent = OpponentModel(
            evaluator=self._evaluator,
            temperature=self.config.opponent_temperature,
        )
        
        self._user = UserModel(
            evaluator=self._evaluator,
            temperature=self.config.user_temperature,
        )
        
        self._initialized = True
        logger.info("CoachTalSelector initialized")
    
    def select(
        self,
        board: chess.Board,
        candidate_scores: Dict[chess.Move, float],
    ) -> SelectionResult:
        """
        Select the best move using cognitive asymmetry optimization.
        
        Args:
            board: Current position (user to move).
            candidate_scores: Dict mapping moves to their MCTS visit proportion
                              or policy probability. Higher = better according to MCTS.
        
        Returns:
            SelectionResult with chosen move and full analysis.
        """
        self._ensure_initialized()
        
        if not self.config.enabled:
            # Bypass: just return the best MCTS move
            return self._bypass_selection(board, candidate_scores)
        
        # Get root value for soundness constraint
        root_value, _ = self._evaluator.evaluate(board)
        
        # Get top-K candidates by MCTS score
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.config.top_k_candidates]
        
        # Analyze each candidate
        analyses: List[MoveCandidate] = []
        for move, mcts_score in sorted_candidates:
            analysis = self._analyze_candidate(board, move, mcts_score, root_value)
            analyses.append(analysis)
        
        # Compute deltas relative to best objective move
        if analyses:
            best_value = max(a.value_after for a in analyses)
            baseline_entropy = analyses[0].opponent_entropy  # Use top MCTS move as baseline
            
            for a in analyses:
                a.value_delta = a.value_after - best_value
                a.entropy_delta = a.opponent_entropy - baseline_entropy
        
        # Select: best J among sound moves, or fallback to MCTS best
        sound_candidates = [a for a in analyses if a.is_sound]
        
        if sound_candidates:
            best = max(sound_candidates, key=lambda a: a.j_score)
            fallback = False
        else:
            # No sound moves pass constraint – fall back to MCTS best
            best = analyses[0] if analyses else None
            fallback = True
            logger.warning("No sound candidates found, falling back to MCTS best")
        
        if best is None:
            # Edge case: no candidates at all
            raise ValueError("No candidate moves provided")
        
        return SelectionResult(
            chosen_move=best.move,
            chosen_analysis=best,
            all_candidates=analyses,
            root_value=root_value,
            fallback_used=fallback,
        )
    
    def _analyze_candidate(
        self,
        board: chess.Board,
        move: chess.Move,
        mcts_score: float,
        root_value: float,
    ) -> MoveCandidate:
        """
        Analyze a single candidate move.
        
        Args:
            board: Current position.
            move: Candidate move to analyze.
            mcts_score: MCTS visit proportion for this move.
            root_value: V_φ of the starting position.
            
        Returns:
            MoveCandidate with all metrics computed.
        """
        # Apply the move to get resulting position
        board_after = board.copy()
        board_after.push(move)
        
        # If game is over after our move, handle specially
        if board_after.is_game_over():
            return self._handle_terminal_candidate(
                move, mcts_score, board_after, root_value
            )
        
        # Get opponent's perspective on the resulting position
        # (it's now opponent's turn)
        opp_entropy = self._opponent.get_entropy(board_after)
        value_after, _ = self._evaluator.evaluate(board_after)
        # Negate value since it's from opponent's perspective
        value_after = -value_after
        
        # For user ease, we'd ideally look at the position after opponent's reply.
        # For v0 simplicity, we approximate by looking at how clear our move was
        # (i.e., was this an "obvious" move for us?).
        # We use the user model on the original position.
        user_ease_score = self._user.get_ease(board)
        
        # Compute J score
        j_score = cognitive_asymmetry_score(
            value=value_after,
            user_ease_score=user_ease_score,
            opponent_entropy=opp_entropy,
            lambda_psych=self.config.lambda_psych,
            gamma_confusion=self.config.gamma_confusion,
        )
        
        # Check soundness
        is_sound = passes_soundness_constraint(
            value_after=value_after,
            value_before=root_value,
            delta=self.config.delta_soundness,
        )
        
        return MoveCandidate(
            move=move,
            mcts_score=mcts_score,
            value_after=value_after,
            opponent_entropy=opp_entropy,
            user_ease=user_ease_score,
            j_score=j_score,
            is_sound=is_sound,
        )
    
    def _handle_terminal_candidate(
        self,
        move: chess.Move,
        mcts_score: float,
        board_after: chess.Board,
        root_value: float,
    ) -> MoveCandidate:
        """Handle a move that ends the game."""
        if board_after.is_checkmate():
            # We delivered checkmate – best possible outcome
            value_after = 1.0
            opp_entropy = 0.0  # No moves for opponent
            user_ease_score = 1.0  # Obvious winning move
        else:
            # Draw (stalemate, repetition, etc.)
            value_after = 0.0
            opp_entropy = 0.0
            user_ease_score = 0.5  # Draw is "okay" but not exciting
        
        j_score = cognitive_asymmetry_score(
            value=value_after,
            user_ease_score=user_ease_score,
            opponent_entropy=opp_entropy,
            lambda_psych=self.config.lambda_psych,
            gamma_confusion=self.config.gamma_confusion,
        )
        
        is_sound = passes_soundness_constraint(
            value_after=value_after,
            value_before=root_value,
            delta=self.config.delta_soundness,
        )
        
        return MoveCandidate(
            move=move,
            mcts_score=mcts_score,
            value_after=value_after,
            opponent_entropy=opp_entropy,
            user_ease=user_ease_score,
            j_score=j_score,
            is_sound=is_sound,
        )
    
    def _bypass_selection(
        self,
        board: chess.Board,
        candidate_scores: Dict[chess.Move, float],
    ) -> SelectionResult:
        """Bypass Coach Tal and just return MCTS best move."""
        if not candidate_scores:
            raise ValueError("No candidate moves provided")
        
        best_move = max(candidate_scores.items(), key=lambda x: x[1])[0]
        
        # Minimal analysis
        root_value, _ = self._evaluator.evaluate(board)
        
        analysis = MoveCandidate(
            move=best_move,
            mcts_score=candidate_scores[best_move],
            value_after=0.0,  # Not computed in bypass mode
            opponent_entropy=0.0,
            user_ease=0.0,
            j_score=0.0,
            is_sound=True,
        )
        
        return SelectionResult(
            chosen_move=best_move,
            chosen_analysis=analysis,
            all_candidates=[analysis],
            root_value=root_value,
            fallback_used=False,
        )
    
    def select_from_board(
        self,
        board: chess.Board,
        top_k: Optional[int] = None,
    ) -> SelectionResult:
        """
        Select a move using only the neural network (no MCTS).
        
        Convenience method that uses the raw policy as candidate scores.
        Useful for quick analysis or when MCTS is not available.
        
        Args:
            board: Position to analyze.
            top_k: Number of top policy moves to consider (default: config.top_k_candidates).
            
        Returns:
            SelectionResult with chosen move and analysis.
        """
        self._ensure_initialized()
        
        _, policy = self._evaluator.evaluate(board)
        
        if top_k is None:
            top_k = self.config.top_k_candidates
        
        # Get top-K by raw policy
        sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidate_scores = dict(sorted_moves)
        
        return self.select(board, candidate_scores)









