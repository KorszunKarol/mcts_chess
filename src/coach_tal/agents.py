"""
Opponent and User model proxies for Coach Tal.

These classes wrap the transformer evaluator to provide cognitive metrics
from the perspective of either the opponent or the user:
    - OpponentModel: Computes H_opp (opponent confusion/entropy)
    - UserModel: Computes E_user (user ease / policy sharpness)

For v0, both use the same underlying neural network. The asymmetry comes
from *which* positions we evaluate (opponent-to-move vs user-to-move) and
how we interpret the metrics (high entropy is bad for opponent, good for us).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import chess

from src.coach_tal.evaluator import TransformerEvaluator
from src.coach_tal.metrics import entropy, user_ease, max_entropy

logger = logging.getLogger(__name__)


@dataclass
class OpponentModel:
    """
    Model of the opponent's decision-making.
    
    Given a position where it's the opponent's turn, computes:
        - H_opp: Shannon entropy of the opponent's policy (confusion)
        - Policy distribution over opponent's legal moves
    
    Higher entropy means the opponent has no clear best move and is more
    likely to make mistakes.
    
    Attributes:
        evaluator: Shared TransformerEvaluator instance.
        temperature: Softmax temperature to simulate bounded rationality.
                     Higher temperature = more uniform policy = weaker player.
    """
    
    evaluator: TransformerEvaluator
    temperature: float = 1.2  # Slightly elevated to model ~1400 Elo mistakes
    
    def __post_init__(self) -> None:
        # Store original temperature to restore after calls
        self._original_temp = self.evaluator.temperature
    
    def get_entropy(self, board: chess.Board) -> float:
        """
        Compute opponent confusion (entropy) for a position.
        
        Args:
            board: Position where it's the opponent's turn to move.
            
        Returns:
            H_opp: Shannon entropy of opponent's policy in nats.
        """
        policy = self.get_policy(board)
        return entropy(policy)
    
    def get_policy(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get the opponent's policy distribution.
        
        Args:
            board: Position where it's the opponent's turn.
            
        Returns:
            Dict mapping legal moves to probabilities.
        """
        # Temporarily adjust temperature
        self.evaluator.temperature = self.temperature
        try:
            _, policy = self.evaluator.evaluate(board)
        finally:
            self.evaluator.temperature = self._original_temp
        
        return policy
    
    def get_entropy_and_value(self, board: chess.Board) -> tuple[float, float]:
        """
        Get both entropy and value for a position.
        
        Useful when you need both metrics without redundant inference.
        
        Args:
            board: Position to evaluate.
            
        Returns:
            Tuple of (entropy, value).
        """
        self.evaluator.temperature = self.temperature
        try:
            value, policy = self.evaluator.evaluate(board)
        finally:
            self.evaluator.temperature = self._original_temp
        
        return entropy(policy), value
    
    def get_normalized_entropy(self, board: chess.Board) -> float:
        """
        Get entropy normalized to [0, 1] range.
        
        0 = opponent has one clear best move
        1 = opponent sees all moves as equally good
        
        Args:
            board: Position to evaluate.
            
        Returns:
            Normalized entropy in [0, 1].
        """
        policy = self.get_policy(board)
        if not policy:
            return 0.0
        
        h = entropy(policy)
        h_max = max_entropy(len(policy))
        
        if h_max == 0:
            return 0.0
        
        return h / h_max


@dataclass
class UserModel:
    """
    Model of the user's decision-making.
    
    Given a position where it's the user's turn, computes:
        - E_user: User ease score (how clear the best move is)
        - Policy distribution over user's legal moves
    
    Higher ease means the user has a clear best move and is less likely
    to go wrong.
    
    Attributes:
        evaluator: Shared TransformerEvaluator instance.
        temperature: Softmax temperature. Lower = sharper policy.
                     For the user model, we use standard temp (1.0) to
                     represent the "ideal" policy the user should follow.
    """
    
    evaluator: TransformerEvaluator
    temperature: float = 1.0  # Standard temperature for user
    
    def __post_init__(self) -> None:
        self._original_temp = self.evaluator.temperature
    
    def get_ease(self, board: chess.Board) -> float:
        """
        Compute user ease for a position.
        
        Args:
            board: Position where it's the user's turn to move.
            
        Returns:
            E_user: Normalized ease score in [0, 1].
                    1.0 = one clear best move
                    0.0 = all moves look equally good
        """
        policy = self.get_policy(board)
        return user_ease(policy)
    
    def get_policy(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get the user's ideal policy distribution.
        
        Args:
            board: Position where it's the user's turn.
            
        Returns:
            Dict mapping legal moves to probabilities.
        """
        self.evaluator.temperature = self.temperature
        try:
            _, policy = self.evaluator.evaluate(board)
        finally:
            self.evaluator.temperature = self._original_temp
        
        return policy
    
    def get_ease_and_value(self, board: chess.Board) -> tuple[float, float]:
        """
        Get both ease and value for a position.
        
        Args:
            board: Position to evaluate.
            
        Returns:
            Tuple of (ease, value).
        """
        self.evaluator.temperature = self.temperature
        try:
            value, policy = self.evaluator.evaluate(board)
        finally:
            self.evaluator.temperature = self._original_temp
        
        return user_ease(policy), value
    
    def get_entropy(self, board: chess.Board) -> float:
        """
        Get raw entropy of user's policy.
        
        Lower entropy = easier position for user.
        
        Args:
            board: Position to evaluate.
            
        Returns:
            Entropy in nats.
        """
        policy = self.get_policy(board)
        return entropy(policy)


def create_agent_pair(
    weights_path: str,
    use_pytorch: bool = False,
    opponent_temperature: float = 1.2,
    user_temperature: float = 1.0,
) -> tuple[OpponentModel, UserModel]:
    """
    Create a matched pair of opponent and user models sharing one evaluator.
    
    This is the recommended way to instantiate the models, as it ensures
    they share the same underlying neural network (efficient memory use).
    
    Args:
        weights_path: Path to transformer weights.
        use_pytorch: Whether to use PyTorch backend.
        opponent_temperature: Temperature for opponent model.
        user_temperature: Temperature for user model.
        
    Returns:
        Tuple of (OpponentModel, UserModel).
    """
    evaluator = TransformerEvaluator(
        weights_path=weights_path,
        use_pytorch=use_pytorch,
        temperature=1.0,  # Base temperature
    )
    
    opponent = OpponentModel(evaluator=evaluator, temperature=opponent_temperature)
    user = UserModel(evaluator=evaluator, temperature=user_temperature)
    
    return opponent, user





