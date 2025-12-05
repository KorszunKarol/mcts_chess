# tests/unit/core/test_evaluator.py

import os
import unittest
import chess
import numpy as np
import tensorflow as tf
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.evaluator import DualHeadEvaluator
from src.move_mapping import move_to_index, ACTION_SPACE_SIZE


class TestDualHeadEvaluator(unittest.TestCase):
    """
    A robust and comprehensive test suite for the DualHeadEvaluator class.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a randomly initialized evaluator once for all tests."""
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

        legal_move_indices = {move_to_index(m, board) for m in board.legal_moves}

        for i in range(ACTION_SPACE_SIZE):
            if i not in legal_move_indices:
                self.assertEqual(
                    policy[i], 0.0, f"Policy for illegal move index {i} should be 0.0."
                )

    def test_board_mirroring_consistency(self):
        """
        Tests that evaluation of a position is consistent regardless of whose turn it is.
        """
        board_b = chess.Board(
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 1"
        )
        board_w = board_b.mirror()

        mock_raw_value = np.array([[0.25]])
        mock_raw_policy = np.random.rand(1, ACTION_SPACE_SIZE).astype(np.float32)

        with mock.patch.object(
            self.evaluator.model,
            "predict",
            return_value=[mock_raw_value, mock_raw_policy],
        ) as mock_predict:
            value_b, _ = self.evaluator.evaluate(board_b)
            value_w, _ = self.evaluator.evaluate(board_w)

            self.assertEqual(mock_predict.call_count, 2)
            self.assertAlmostEqual(value_b, value_w, places=5)
            self.assertAlmostEqual(value_b, 0.25, places=5)

    def test_terminal_node_evaluation(self):
        """Tests evaluation on a checkmated or stalemate board."""
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        board.push_uci("d1h5")
        board.push_uci("b8c6")
        board.push_uci("f1c4")
        board.push_uci("g8f6")
        board.push_uci("h5f7")

        value, policy = self.evaluator.evaluate(board)
        self.assertTrue(-1.0 <= value <= 1.0)
        total = np.sum(policy)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            self.assertAlmostEqual(total, 1.0, places=5)
        else:
            self.assertAlmostEqual(total, 0.0, places=5)

    def test_policy_shape_matches_action_space(self):
        """Ensure policy output matches expected action space size."""
        board = chess.Board()
        _, policy = self.evaluator.evaluate(board)
        self.assertEqual(policy.shape[0], ACTION_SPACE_SIZE)


if __name__ == "__main__":
    unittest.main()

