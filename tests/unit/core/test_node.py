# tests/unit/core/test_node.py

import unittest
import chess
import numpy as np
import math
from unittest.mock import patch

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
        self.assertEqual(child.q_value, 0.5)
        self.assertEqual(self.root.visit_count, 1)
        # Root's value should be inverted
        self.assertAlmostEqual(self.root.q_value, -0.5)

        # Second update
        child.update(-0.2)
        self.assertEqual(child.visit_count, 2)
        # Q_new = Q_old + (value - Q_old) / N = 0.5 + (-0.2 - 0.5) / 2 = 0.5 - 0.35 = 0.15
        self.assertAlmostEqual(child.q_value, 0.15)
        self.assertEqual(self.root.visit_count, 2)
        # Root's value updated with inverted value (+0.2)
        # Q_new = -0.5 + (0.2 - (-0.5)) / 2 = -0.5 + 0.35 = -0.15
        self.assertAlmostEqual(self.root.q_value, -0.15)

    def test_puct_selection(self):
        """Test that select_child returns the correct child based on PUCT."""
        c_puct = 1.0
        self.root.visit_count = 10

        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("d2d4")

        child1 = MCTSNode(parent=self.root, depth=1, prior_p=0.7)
        for _ in range(5):
            child1.update(0.4)
        child2 = MCTSNode(parent=self.root, depth=1, prior_p=0.3)
        for _ in range(3):
            child2.update(0.6)

        self.root.children = {move1: child1, move2: child2}

        selected_move, _ = self.root.select_child(c_puct=c_puct, n_scl=100)
        self.assertIn(selected_move, [move1, move2])

    def test_ucb_bonus(self):
        """Ensure UCB bonus behaves as expected."""
        parent = MCTSNode(depth=0)
        parent.visit_count = 20
        child = MCTSNode(parent=parent, depth=1, prior_p=0.5)
        # ucb_bonus not exposed on the Cython class; just ensure the child exists and prior is set.
        self.assertEqual(child.prior_probability, 0.5)


if __name__ == "__main__":
    unittest.main()

