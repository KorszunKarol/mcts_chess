# src/tests/test_node.py

import unittest
import chess
import numpy as np
import math
from unittest.mock import patch

# Assuming the MCTSNode class is in this path
from src.mcts.node import MCTSNode


class TestMCTSNode(unittest.TestCase):

    def setUp(self):
        """Set up a root node before each test."""
        self.root = MCTSNode(depth=0)

    def test_initialization(self):
        """Test that all attributes are initialized correctly."""
        self.assertIsNone(self.root.parent)
        self.assertEqual(self.root.children, {})
        self.assertEqual(self.root.visit_count, 0)
        self.assertEqual(self.root.mean_action_value, 0.0)
        self.assertEqual(self.root.q_value, 0.0)
        self.assertEqual(self.root.prior_probability, 1.0)
        self.assertEqual(self.root.depth, 0)
        self.assertFalse(self.root.is_frozen)
        self.assertIsNone(self.root.frozen_visit_counts)
        self.assertTrue(self.root.is_leaf())

    def test_expand(self):
        """Test node expansion with a sample policy."""
        board = chess.Board()
        moves = list(board.legal_moves)
        policy = {moves[0]: 0.6, moves[1]: 0.4}

        self.root.expand(policy)
        self.assertFalse(self.root.is_leaf())
        self.assertEqual(len(self.root.children), 2)

        child1 = self.root.children[moves[0]]
        self.assertIs(child1.parent, self.root)
        self.assertEqual(child1.prior_probability, 0.6)
        self.assertEqual(child1.depth, 1)

    def test_update_and_q_value(self):
        """Test the backpropagation (update) logic and Q-value calculation."""
        child = MCTSNode(parent=self.root, depth=1)
        self.root.children[chess.Move.from_uci("e2e4")] = child

        # First update
        child.update(0.5)
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(child.q_value, 0.5)  # Corrected: Access as property
        self.assertEqual(self.root.visit_count, 1)
        # Root's value should be inverted
        self.assertAlmostEqual(self.root.q_value, -0.5)

        # Second update
        child.update(-0.2)
        self.assertEqual(child.visit_count, 2)
        # Q_new = Q_old + (value - Q_old) / N = 0.5 + (-0.2 - 0.5) / 2 = 0.5 - 0.35 = 0.15
        self.assertAlmostEqual(child.q_value, 0.15)  # Corrected: Access as property
        self.assertEqual(self.root.visit_count, 2)
        # Root's value updated with inverted value (+0.2)
        # Q_new = -0.5 + (0.2 - (-0.5)) / 2 = -0.5 + 0.35 = -0.15
        self.assertAlmostEqual(self.root.q_value, -0.15)

    def test_puct_selection(self):
        """Test that select_child returns the correct child based on PUCT."""
        c_puct = 1.0
        self.root.visit_count = 10

        move1 = chess.Move.from_uci("e2e4")
        child1 = MCTSNode(parent=self.root, prior_p=0.6)
        child1.visit_count = 5
        child1.mean_action_value = 0.5
        self.root.children[move1] = child1

        move2 = chess.Move.from_uci("d2d4")
        child2 = MCTSNode(parent=self.root, prior_p=0.4)
        child2.visit_count = 2
        child2.mean_action_value = -0.2
        self.root.children[move2] = child2

        # Manually verify scores
        score1 = 0.5 + c_puct * 0.6 * (math.sqrt(10) / (1 + 5))
        score2 = -0.2 + c_puct * 0.4 * (math.sqrt(10) / (1 + 2))
        self.assertGreater(score1, score2)

        selected_move, _ = self.root.select_child(c_puct=c_puct, n_scl=100)
        self.assertEqual(selected_move, move1)

    # --- New Edge Case Tests ---

    def test_select_child_on_leaf_node(self):
        """CRITICAL EDGE CASE: Test that calling select_child on a leaf node fails gracefully."""
        self.assertTrue(self.root.is_leaf())
        result = self.root.select_child(c_puct=1.0, n_scl=10)
        self.assertIsNone(result, "select_child on a leaf node should return None")

    def test_puct_with_zero_parent_visits(self):
        """
        EDGE CASE: Test PUCT calculation stability when parent has zero visits.
        The exploration term should be zero.
        """
        self.assertEqual(self.root.visit_count, 0)  # Parent visits = 0

        move1 = chess.Move.from_uci("e2e4")
        child1 = MCTSNode(parent=self.root, prior_p=0.6)
        child1.mean_action_value = 0.8  # High Q
        self.root.children[move1] = child1

        move2 = chess.Move.from_uci("d2d4")
        child2 = MCTSNode(parent=self.root, prior_p=0.4)
        child2.mean_action_value = 0.9  # Even higher Q
        self.root.children[move2] = child2

        # With N_parent = 0, PUCT score just becomes Q. Child 2 should be selected.
        selected_move, _ = self.root.select_child(c_puct=1.0, n_scl=100)
        self.assertEqual(
            selected_move,
            move2,
            "With zero parent visits, selection should be purely based on Q-value.",
        )

    def test_search_contempt_logic(self):
        """Test the freezing and Thompson sampling logic for opponent nodes."""
        opponent_node = MCTSNode(depth=1)
        opponent_node.visit_count = 15
        n_scl = 10
        c_puct = 1.0

        move1 = chess.Move.from_uci("e7e5")
        child1 = MCTSNode(parent=opponent_node, prior_p=0.8, depth=2)
        opponent_node.children[move1] = child1

        move2 = chess.Move.from_uci("c7c5")
        child2 = MCTSNode(parent=opponent_node, prior_p=0.2, depth=2)
        opponent_node.children[move2] = child2

        # 1. Test PUCT selection when total visits are below n_scl
        child1.visit_count = 5
        child1.mean_action_value = 0.1
        child2.visit_count = 4
        child2.mean_action_value = 0.0

        self.assertFalse(opponent_node.is_frozen)
        selected_move, _ = opponent_node.select_child(c_puct, n_scl)
        self.assertEqual(
            selected_move,
            move1,
            "Should select child with higher PUCT before freezing.",
        )

        # 2. Test freezing when visits exceed n_scl
        child2.visit_count = 6  # Total visits = 5 + 6 = 11 > 10

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = move2

            selected_move, _ = opponent_node.select_child(c_puct, n_scl)

            self.assertTrue(opponent_node.is_frozen)
            self.assertEqual(opponent_node.frozen_visit_counts, {move1: 5, move2: 6})
            mock_choice.assert_called_once()
            args, kwargs = mock_choice.call_args

            # Check probabilities passed to sampler
            probs = kwargs["p"]
            expected_probs = np.array([5, 6], dtype=np.float32) / 11
            np.testing.assert_allclose(probs, expected_probs)
            self.assertEqual(selected_move, move2)

        # 3. Test that subsequent calls use the already frozen distribution
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = move1

            selected_move, _ = opponent_node.select_child(c_puct, n_scl)

            # Should still be frozen with original counts
            self.assertEqual(opponent_node.frozen_visit_counts, {move1: 5, move2: 6})
            mock_choice.assert_called_once()
            self.assertEqual(selected_move, move1)


if __name__ == "__main__":
    unittest.main()
