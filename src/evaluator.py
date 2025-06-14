from dataclasses import dataclass
import chess
import numpy as np
import tensorflow as tf
from typing import List, Union, Tuple, Dict
from src.encoder import Encoder
from src.model import create_model
from src.move_mapping import move_to_index, index_to_move, ACTION_SPACE_SIZE
from src.utils import unmirror_policy
import logging

logger = logging.getLogger(__name__)


@dataclass
class DualHeadEvaluator:
    """
    Wraps the dual-head neural network to provide evaluations of board positions.

    This class acts as a bridge between the chess domain (chess.Board objects)
    and the neural network (TensorFlow tensors). It handles encoding, prediction,
    and decoding, including the crucial step of masking illegal moves in the
    policy output.
    """

    weights_path: str

    def __post_init__(self):
        logger.debug(f"Loading model weights from {self.weights_path}")
        self.model = create_model()
        if self.weights_path:
            try:
                self.model.load_weights(self.weights_path)
                logger.debug("Model weights loaded successfully")
            except Exception as e:
                logger.error(
                    f"Could not load model weights from {self.weights_path}: {e}"
                )
                # Depending on requirements, you might want to raise the exception
                # or proceed with an uninitialized model for things like training.
        self.encoder = Encoder()

    def evaluate(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """
        Evaluates a single board position using board mirroring for black.

        Args:
            board: The chess.Board object to evaluate.

        Returns:
            A tuple containing:
            - The value of the position (from the perspective of the current player).
            - A numpy array of shape (ACTION_SPACE_SIZE,) representing the
              probability distribution over all possible moves.
        """
        if board.is_game_over():
            return self._handle_game_over(board), np.zeros(ACTION_SPACE_SIZE)

        # White's turn: evaluate directly.
        # Black's turn: mirror the board, evaluate, then un-mirror the policy.
        if board.turn == chess.WHITE:
            logger.debug("White's turn. Evaluating board directly.")
            position_tensor = self.encoder.encode(board)
            tensor = np.expand_dims(position_tensor, axis=0)
            value_pred, policy_logits = self.model.predict(tensor, verbose=0)
            policy_logits = policy_logits[0]
        else:  # Black's turn
            logger.debug("Black's turn. Mirroring board for evaluation.")
            mirrored_board = board.mirror()
            position_tensor = self.encoder.encode(mirrored_board)
            tensor = np.expand_dims(position_tensor, axis=0)
            value_pred, raw_policy_logits = self.model.predict(tensor, verbose=0)
            raw_policy_logits = raw_policy_logits[0]

            logger.debug(f"Mirrored policy logits (top 5): {raw_policy_logits[:5]}")
            policy_logits = unmirror_policy(raw_policy_logits)
            logger.debug(f"Un-mirrored policy logits (top 5): {policy_logits[:5]}")

        # Squeeze batch dimension from results
        value = float(value_pred[0][0])

        # The mask is always created from the perspective of the original board
        legal_moves = list(board.legal_moves)
        legal_move_indices = [
            move_to_index(m) for m in legal_moves if move_to_index(m) is not None
        ]

        # Create a mask for legal moves, setting illegal moves to -inf
        mask = np.full(policy_logits.shape, -np.inf)
        mask[legal_move_indices] = 0

        # Apply the mask and compute softmax to get a probability distribution
        masked_logits = policy_logits + mask
        probabilities = tf.nn.softmax(masked_logits).numpy()

        # Log final evaluation result
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_moves = [
            (index_to_move(i).uci(), probabilities[i])
            for i in top_3_indices
            if index_to_move(i)
        ]
        logger.debug(f"Final evaluation: Value={value:.4f}, Top 3 moves: {top_3_moves}")

        return value, probabilities

    def _handle_game_over(self, board: chess.Board) -> float:
        """Returns the definitive value of a game-over position."""
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            # From white's perspective, so -1.0 is correct
            return -1.0
        return 0.0

    def save_model(self, path: str):
        """Save the full model architecture and weights."""
        self.model.save(path)

    @classmethod
    def load_full_model(cls, model_path: str):
        """Load a full model (architecture + weights)."""
        evaluator = cls(weights_path="")  # Dummy path
        evaluator.model = tf.keras.models.load_model(model_path)
        logger.info(f"Full model loaded successfully from {model_path}")
        return evaluator
