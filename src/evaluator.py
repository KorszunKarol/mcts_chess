"""
Lightweight evaluator stub used by tests.

Provides a CPU-only DualHeadEvaluator that returns a value in [-1, 1] and a
legal-move-masked policy over the ACTION_SPACE_SIZE defined in move_mapping.
This is a minimal replacement when full model weights are unavailable.
"""

from __future__ import annotations

import numpy as np
import chess
import tensorflow as tf

from src.move_mapping import ACTION_SPACE_SIZE, move_to_index


class DualHeadEvaluator:
    def __init__(self, weights_path: str | None = None):
        # Minimal stub model; tests may mock .predict
        inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
        zeros = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=[zeros, zeros])
        self.weights_path = weights_path

    def _encode_board(self, board: chess.Board) -> np.ndarray:
        # Simple scalar encoding; structure is irrelevant for current tests
        return np.array([[float(board.fullmove_number)]], dtype=np.float32)

    def evaluate(self, board: chess.Board) -> tuple[float, np.ndarray]:
        # Use model.predict to allow mocking in tests
        encoded = self._encode_board(board)
        raw_value, raw_policy = self.model.predict(encoded, verbose=0)

        value = float(np.clip(raw_value[0][0], -1.0, 1.0))

        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            # If model produced a policy vector, use it; otherwise uniform
            if raw_policy.size >= ACTION_SPACE_SIZE:
                base = np.array(raw_policy[0][:ACTION_SPACE_SIZE], dtype=np.float32)
                base = np.maximum(base, 0.0)
                for move in legal_moves:
                    idx = move_to_index(move, board)
                    if idx is not None:
                        policy[idx] = base[idx]
            else:
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    idx = move_to_index(move, board)
                    if idx is not None:
                        policy[idx] = prob

            total = policy.sum()
            if total > 0:
                policy /= total

        return value, policy


__all__ = ["DualHeadEvaluator"]

