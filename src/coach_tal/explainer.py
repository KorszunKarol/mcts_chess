"""
Natural language explainer for Coach Tal move recommendations.

This module generates human-readable explanations for why Coach Tal
selected a particular move, based on the cognitive asymmetry metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import chess

from src.coach_tal.selector import MoveCandidate, SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """
    Structured analysis of a move for display.
    
    Contains both raw metrics and human-readable explanations.
    """
    
    move: chess.Move
    move_san: str  # Standard algebraic notation
    
    # Raw metrics
    value: float
    opponent_entropy: float
    user_ease: float
    j_score: float
    is_sound: bool
    
    # Comparative metrics
    value_delta: float  # vs best objective move
    entropy_delta: float  # vs baseline
    
    # Explanations
    primary_reason: str
    secondary_notes: List[str]
    
    # Classification
    move_type: str  # "sharp", "solid", "tricky", "forced", etc.


class Explainer:
    """
    Generates explanations for Coach Tal move selections.
    
    Takes a SelectionResult and produces human-readable analysis
    suitable for display in a CLI or GUI.
    
    Example:
        explainer = Explainer()
        result = selector.select(board, mcts_policy)
        analysis = explainer.explain(result, board)
        print(analysis.primary_reason)
    """
    
    # Thresholds for classification
    HIGH_ENTROPY_THRESHOLD = 2.5  # nats (~12 equally likely moves)
    LOW_ENTROPY_THRESHOLD = 1.0   # nats (~3 equally likely moves)
    HIGH_EASE_THRESHOLD = 0.7
    LOW_EASE_THRESHOLD = 0.3
    SIGNIFICANT_VALUE_DELTA = 0.1
    SIGNIFICANT_ENTROPY_DELTA = 0.5
    
    def explain(
        self,
        result: SelectionResult,
        board: chess.Board,
    ) -> MoveAnalysis:
        """
        Generate explanation for the chosen move.
        
        Args:
            result: SelectionResult from CoachTalSelector.
            board: The position before the move.
            
        Returns:
            MoveAnalysis with explanations.
        """
        chosen = result.chosen_analysis
        
        # Get SAN notation
        move_san = board.san(chosen.move)
        
        # Classify the move
        move_type = self._classify_move(chosen, result)
        
        # Generate primary reason
        primary_reason = self._generate_primary_reason(chosen, result, move_type)
        
        # Generate secondary notes
        secondary_notes = self._generate_secondary_notes(chosen, result)
        
        return MoveAnalysis(
            move=chosen.move,
            move_san=move_san,
            value=chosen.value_after,
            opponent_entropy=chosen.opponent_entropy,
            user_ease=chosen.user_ease,
            j_score=chosen.j_score,
            is_sound=chosen.is_sound,
            value_delta=chosen.value_delta,
            entropy_delta=chosen.entropy_delta,
            primary_reason=primary_reason,
            secondary_notes=secondary_notes,
            move_type=move_type,
        )
    
    def explain_all(
        self,
        result: SelectionResult,
        board: chess.Board,
    ) -> List[MoveAnalysis]:
        """
        Generate explanations for all candidate moves.
        
        Args:
            result: SelectionResult from CoachTalSelector.
            board: The position before the move.
            
        Returns:
            List of MoveAnalysis for all candidates, sorted by J score.
        """
        analyses = []
        
        for candidate in sorted(
            result.all_candidates,
            key=lambda c: c.j_score,
            reverse=True,
        ):
            # Create a temporary result for this candidate
            temp_result = SelectionResult(
                chosen_move=candidate.move,
                chosen_analysis=candidate,
                all_candidates=result.all_candidates,
                root_value=result.root_value,
                fallback_used=False,
            )
            
            analysis = self.explain(temp_result, board)
            analyses.append(analysis)
        
        return analyses
    
    def _classify_move(
        self,
        chosen: MoveCandidate,
        result: SelectionResult,
    ) -> str:
        """Classify the move into a strategic category."""
        h_opp = chosen.opponent_entropy
        e_user = chosen.user_ease
        v_delta = chosen.value_delta
        
        # Check for forcing/winning moves
        if chosen.value_after > 0.8:
            return "winning"
        
        if chosen.value_after < -0.5 and not chosen.is_sound:
            return "desperate"
        
        # High opponent entropy = tricky
        if h_opp > self.HIGH_ENTROPY_THRESHOLD:
            if e_user > self.HIGH_EASE_THRESHOLD:
                return "tricky"  # Hard for them, easy for us
            else:
                return "sharp"  # Hard for everyone
        
        # Low opponent entropy = solid
        if h_opp < self.LOW_ENTROPY_THRESHOLD:
            if abs(v_delta) < self.SIGNIFICANT_VALUE_DELTA:
                return "solid"
            else:
                return "principled"
        
        # High user ease = intuitive
        if e_user > self.HIGH_EASE_THRESHOLD:
            return "intuitive"
        
        # Default
        return "balanced"
    
    def _generate_primary_reason(
        self,
        chosen: MoveCandidate,
        result: SelectionResult,
        move_type: str,
    ) -> str:
        """Generate the main explanation for the move choice."""
        h_opp = chosen.opponent_entropy
        e_user = chosen.user_ease
        v = chosen.value_after
        v_delta = chosen.value_delta
        
        # Handle special cases first
        if result.fallback_used:
            return "This is the objectively best move. Other options were too risky."
        
        if move_type == "winning":
            return "This move wins material or delivers a decisive advantage."
        
        if move_type == "desperate":
            return "The position is difficult. This creates the most problems for your opponent."
        
        # Build explanation based on metrics
        reasons = []
        
        # Value-based reasoning
        if abs(v_delta) < 0.05:
            reasons.append("objectively equal to the alternatives")
        elif v_delta > 0:
            reasons.append("slightly better objectively")
        else:
            reasons.append("a small concession objectively")
        
        # Entropy-based reasoning
        if h_opp > self.HIGH_ENTROPY_THRESHOLD:
            reasons.append("gives your opponent many tempting but unclear options")
        elif h_opp < self.LOW_ENTROPY_THRESHOLD:
            reasons.append("keeps the position simple and controlled")
        
        # User ease reasoning
        if e_user > self.HIGH_EASE_THRESHOLD:
            reasons.append("your follow-up moves will be natural and intuitive")
        elif e_user < self.LOW_EASE_THRESHOLD:
            reasons.append("requires careful calculation in the follow-up")
        
        # Combine into a sentence
        if move_type == "tricky":
            return f"This move is {reasons[0]}, but {reasons[1] if len(reasons) > 1 else 'creates practical problems'}."
        elif move_type == "solid":
            return f"A reliable choice that {reasons[0]}."
        elif move_type == "sharp":
            return f"A double-edged move: {', '.join(reasons[:2])}."
        elif move_type == "intuitive":
            return f"The natural move here – {reasons[0]}."
        else:
            return f"This move {reasons[0]}."
    
    def _generate_secondary_notes(
        self,
        chosen: MoveCandidate,
        result: SelectionResult,
    ) -> List[str]:
        """Generate additional notes about the move."""
        notes = []
        
        # Compare to other candidates
        better_objective = [
            c for c in result.all_candidates
            if c.value_after > chosen.value_after + 0.05 and c.move != chosen.move
        ]
        
        if better_objective:
            best_alt = max(better_objective, key=lambda c: c.value_after)
            notes.append(
                f"Note: {best_alt.move.uci()} is slightly better objectively "
                f"(+{best_alt.value_after - chosen.value_after:.2f}) but harder to play."
            )
        
        # Opponent entropy insight
        if chosen.opponent_entropy > self.HIGH_ENTROPY_THRESHOLD:
            notes.append(
                "Your opponent will likely spend significant time here, "
                "increasing their chance of error."
            )
        
        # User ease insight
        if chosen.user_ease > self.HIGH_EASE_THRESHOLD:
            notes.append(
                "Your next moves should flow naturally from here."
            )
        elif chosen.user_ease < self.LOW_EASE_THRESHOLD:
            notes.append(
                "Be prepared to calculate carefully after this move."
            )
        
        # Soundness warning
        if not chosen.is_sound:
            notes.append(
                "⚠️ This move may be objectively dubious but creates practical chances."
            )
        
        return notes
    
    def format_summary(
        self,
        analysis: MoveAnalysis,
        verbose: bool = False,
    ) -> str:
        """
        Format analysis as a displayable string.
        
        Args:
            analysis: MoveAnalysis to format.
            verbose: Include all details if True.
            
        Returns:
            Formatted string for display.
        """
        lines = []
        
        # Header
        lines.append(f"► {analysis.move_san} ({analysis.move_type})")
        lines.append(f"  {analysis.primary_reason}")
        
        if verbose:
            # Metrics
            lines.append("")
            lines.append(f"  Metrics:")
            lines.append(f"    Value: {analysis.value:+.2f} (Δ {analysis.value_delta:+.2f})")
            lines.append(f"    Opponent confusion: {analysis.opponent_entropy:.2f} nats")
            lines.append(f"    Your ease: {analysis.user_ease:.1%}")
            lines.append(f"    J-score: {analysis.j_score:.3f}")
            
            # Secondary notes
            if analysis.secondary_notes:
                lines.append("")
                for note in analysis.secondary_notes:
                    lines.append(f"  • {note}")
        
        return "\n".join(lines)
    
    def format_comparison(
        self,
        analyses: List[MoveAnalysis],
        top_n: int = 3,
    ) -> str:
        """
        Format a comparison of top moves.
        
        Args:
            analyses: List of MoveAnalysis (should be sorted by J score).
            top_n: Number of moves to show.
            
        Returns:
            Formatted comparison string.
        """
        lines = []
        lines.append("Coach Tal's Analysis:")
        lines.append("=" * 40)
        
        for i, analysis in enumerate(analyses[:top_n]):
            rank = i + 1
            marker = "★" if rank == 1 else " "
            
            lines.append(f"\n{marker} #{rank}: {analysis.move_san}")
            lines.append(f"   Type: {analysis.move_type}")
            lines.append(f"   J-score: {analysis.j_score:.3f}")
            lines.append(f"   {analysis.primary_reason}")
            
            if rank == 1 and analysis.secondary_notes:
                for note in analysis.secondary_notes[:1]:  # Just first note
                    lines.append(f"   → {note}")
        
        lines.append("")
        lines.append("=" * 40)
        
        return "\n".join(lines)






