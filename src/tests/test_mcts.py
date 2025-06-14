import unittest
from unittest.mock import Mock, MagicMock
import chess
import numpy as np
from typing import Dict

from src.mcts.worker import MCTS


class MockEvaluator:
    """A mock evaluator that returns predictable values, for testing."""

    def evaluate(self, board: chess.Board) -> tuple[float, Dict[chess.Move, float]]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, {}

        # Create a dummy uniform policy and a fixed value
        dummy_policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
        dummy_value = 0.5

        return dummy_value, dummy_policy


class TestMCTS(unittest.TestCase):

    def setUp(self):
        """Set up a standard MCTS instance before each test."""
        self.mock_evaluator = MockEvaluator()
        self.mcts = MCTS(evaluator=self.mock_evaluator, c_puct=1.0, n_scl=100)
        self.start_board = chess.Board()

    def test_run_search_on_start_pos(self):
        """
        Happy Path: Test a basic search on the starting position.
        """
        policy = self.mcts.run_search(self.start_board, num_simulations=10)

        self.assertIsInstance(policy, dict)
        self.assertGreater(len(policy), 0, "Policy should not be empty.")
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)
        legal_moves = list(self.start_board.legal_moves)
        self.assertTrue(all(move in legal_moves for move in policy.keys()))

    def test_search_on_checkmated_position(self):
        """
        Edge Case: Test search on a terminal (checkmate) position.
        """
        board = chess.Board("4N3/6Qk/8/2pP4/2P5/pP3PNP/P6P/R4RK1 b - - 4 31")
        self.assertTrue(board.is_checkmate())
        self.mock_evaluator.evaluate = MagicMock()

        policy = self.mcts.run_search(board, num_simulations=1)

        self.assertEqual(policy, {})
        self.mock_evaluator.evaluate.assert_not_called()

    def test_search_on_stalemate_position(self):
        """
        Edge Case: Test search on a terminal (stalemate) position.
        """
        # *** FIX: Use a correct FEN for a stalemate position ***
        board = chess.Board("4N2k/8/6Q1/2pP4/2P5/pP3PNP/P6P/R4RK1 b - - 2 30")
        self.assertTrue(board.is_stalemate())

        self.mock_evaluator.evaluate = MagicMock()
        policy = self.mcts.run_search(board, num_simulations=1)

        self.assertEqual(policy, {})
        self.mock_evaluator.evaluate.assert_not_called()

    def test_evaluator_is_called_on_leaf_node(self):
        """
        Verify that the evaluator is correctly called during the expansion phase.
        """
        test_value = 0.75
        test_policy = {chess.Move.from_uci("e2e4"): 1.0}

        self.mock_evaluator.evaluate = Mock(return_value=(test_value, test_policy))
        self.mcts.run_search(self.start_board, num_simulations=1)
        self.mock_evaluator.evaluate.assert_called_once()

    def test_backpropagation_updates_root_correctly(self):
        """
        Check that the value from evaluation is correctly propagated to the root.
        This is better tested in test_node.py, but this confirms the search runs.
        """
        self.mock_evaluator.evaluate = Mock(return_value=(0.5, {}))
        policy = self.mcts.run_search(self.start_board, num_simulations=1)
        self.assertIsInstance(policy, dict)


if __name__ == "__main__":
    unittest.main()
