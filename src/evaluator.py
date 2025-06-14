# src/evaluator.py

from dataclasses import dataclass
import chess
import numpy as np
import tensorflow as tf
from typing import Tuple
from src.encoder import Encoder
from src.model import create_model
from src.move_mapping import move_to_index, ACTION_SPACE_SIZE
from src.utils import unmirror_policy
import logging

logger = logging.getLogger(__name__)


@dataclass
class DualHeadEvaluator:
    """
    Wraps the dual-head neural network to provide evaluations of board positions,
    correctly handling player perspective via board mirroring.
    """

    weights_path: str

    def __post_init__(self):
        logger.debug(f"Loading model from {self.weights_path}")
        self.model = create_model()
        if self.weights_path:
            try:
                self.model.load_weights(self.weights_path)
                logger.debug("Model weights loaded successfully")
            except Exception as e:
                logger.error(
                    f"Could not load model weights from {self.weights_path}: {e}"
                )
        self.encoder = Encoder()

    def evaluate(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """
        Evaluates a single board position, using board mirroring for Black's turn.
        """
        if board.is_game_over(claim_draw=True):
            # *** FIX is applied in the _handle_game_over method ***
            return self._handle_game_over(board), np.zeros(ACTION_SPACE_SIZE)

        # --- Board Mirroring Logic ---
        if board.turn == chess.WHITE:
            encoded_board = self.encoder.encode(board)
            tensor = np.expand_dims(encoded_board, axis=0)
            value_pred, policy_logits = self.model.predict(tensor, verbose=0)
            final_policy_logits = policy_logits[0]
        else:  # Black's turn
            mirrored_board = board.mirror()
            encoded_board = self.encoder.encode(mirrored_board)
            tensor = np.expand_dims(encoded_board, axis=0)
            value_pred, policy_logits = self.model.predict(tensor, verbose=0)
            final_policy_logits = unmirror_policy(policy_logits[0])

        value = float(value_pred[0][0])

        # --- Masking Logic ---
        legal_moves = list(board.legal_moves)

        legal_move_indices = [
            move_to_index(m, board)
            for m in legal_moves
            if move_to_index(m, board) is not None
        ]

        if not legal_move_indices:
            return value, np.zeros(ACTION_SPACE_SIZE)

        mask = np.full(final_policy_logits.shape, -np.inf)
        mask[legal_move_indices] = 0
        masked_logits = final_policy_logits + mask

        probabilities = tf.nn.softmax(masked_logits).numpy()
        if np.isnan(probabilities).any():
            return value, np.zeros(ACTION_SPACE_SIZE)

        return value, probabilities

    def _handle_game_over(self, board: chess.Board) -> float:
        """
        Returns the definitive value of a game-over position from the
        perspective of the current player.
        """
        result = board.result(claim_draw=True)

        if board.is_checkmate():
            # If it's the current player's turn, they are mated and have lost.
            return -1.0

        # For draws (stalemate, insufficient material, etc.), the value is 0.
        return 0.0

    def save_model(self, path: str):
        self.model.save(path)

    @classmethod
    def load_full_model(cls, model_path: str):
        evaluator = cls(weights_path="")
        evaluator.model = tf.keras.models.load_model(model_path)
        logger.info(f"Full model loaded successfully from {model_path}")
        return evaluator
