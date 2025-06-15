# tests/test_evaluator.py

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import unittest
import chess
import numpy as np
import tensorflow as tf
from src.evaluator import DualHeadEvaluator
from src.move_mapping import move_to_index, ACTION_SPACE_SIZE
from unittest import mock

tf.keras.

class TestDualHeadEvaluator(unittest.TestCase):
    """
    A robust and comprehensive test suite for the DualHeadEvaluator class,
    updated with the correct function signatures.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a randomly initialized evaluator once for all tests."""

        # reset the tf devices
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        cls.evaluator = DualHeadEvaluator(weights_path="")
        tf.get_logger().setLevel("ERROR")

    def test_initialization(self):
        """Tests that the evaluator and its model instantiate correctly."""
        self.assertIsInstance(self.evaluator, DualHeadEvaluator)
        self.assertIsInstance(self.evaluator.model, tf.keras.Model)

    def test_evaluate_output_properties(self):
        """Tests the output types, shapes, and value ranges from evaluate()."""
        board = chess.Board()
        value, policy = self.evaluator.evaluate(board)
        self.assertIsInstance(value, float)
        self.assertTrue(
            -1.0 <= value <= 1.0,
            "Value from model should be in [-1, 1], as it's not a terminal node.",
        )
        self.assertIsInstance(policy, np.ndarray)
        self.assertEqual(policy.shape, (ACTION_SPACE_SIZE,))
        self.assertAlmostEqual(np.sum(policy), 1.0, places=5)

    def test_illegal_move_masking(self):
        """Ensures probabilities for illegal moves are zero."""
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        _, policy = self.evaluator.evaluate(board)

        # Pass the 'board' object to move_to_index
        legal_move_indices = {move_to_index(m, board) for m in board.legal_moves}

        for i in range(ACTION_SPACE_SIZE):
            if i not in legal_move_indices:
                self.assertEqual(
                    policy[i], 0.0, f"Policy for illegal move index {i} should be 0.0."
                )

    def test_board_mirroring_consistency(self):
        """
        Tests that the evaluation of a position is consistent regardless of whose
        turn it is, thanks to the board mirroring logic.
        """
        # A position where it is Black's turn
        board_b = chess.Board(
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 1"
        )
        # The exact same position, but from White's perspective
        board_w = board_b.mirror()

        # Mock the model's output to be predictable for a consistent test
        mock_raw_value = np.array([[0.25]])
        mock_raw_policy = np.random.rand(1, ACTION_SPACE_SIZE).astype(np.float32)

        with mock.patch.object(
            self.evaluator.model,
            "predict",
            return_value=[mock_raw_value, mock_raw_policy],
        ) as mock_predict:
            value_b, _ = self.evaluator.evaluate(board_b)
            value_w, _ = self.evaluator.evaluate(board_w)

            # The evaluator should have called predict twice
            self.assertEqual(mock_predict.call_count, 2)

            # The final values should be identical because the mirrored black board
            # becomes the white board, and the model's raw output is the same for both.
            self.assertAlmostEqual(
                value_b,
                value_w,
                places=5,
                msg="Evaluation value should be consistent for a position and its mirror.",
            )
            self.assertAlmostEqual(value_b, 0.25, places=5)

    def test_terminal_node_evaluation(self):
        """Tests evaluation on a checkmated or stalemate board."""
        # *** FIX: Use a correct, legal sequence to reach Scholar's Mate. ***
        board = chess.Board()
        # 1. e4 e5
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        # 2. Qh5 Nc6
        board.push_uci("d1h5")
        board.push_uci("b8c6")
        # 3. Bc4 Nf6?? (A blunder)
        board.push_uci("f1c4")
        board.push_uci("g8f6")
        # 4. Qxf7#
        board.push_uci("h5f7")

        self.assertTrue(board.is_checkmate(), "Board should be in a checkmate state.")

        # It is now Black's turn, and Black is mated.
        value, policy = self.evaluator.evaluate(board)

        # The result of the board is 1-0 (White wins).
        # Since it is Black's turn, the value should be -1.0
        self.assertAlmostEqual(
            value,
            -1.0,
            places=5,
            msg="Value for a checkmated position (for the mated player) should be -1.0.",
        )

        self.assertAlmostEqual(
            np.sum(policy),
            0.0,
            places=5,
            msg="Sum of policy for a terminal node should be 0.0 as there are no legal moves.",
        )


if __name__ == "__main__":
    unittest.main()
