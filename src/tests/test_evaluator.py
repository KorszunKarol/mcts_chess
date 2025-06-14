# src/tests/test_evaluator.py

import unittest
import chess
import numpy as np
import tensorflow as tf
from src.evaluator import DualHeadEvaluator
from src.move_mapping import move_to_index, index_to_move, ACTION_SPACE_SIZE
from src.utils import unmirror_policy


class TestDualHeadEvaluator(unittest.TestCase):
    """
    A more robust and comprehensive test suite for the DualHeadEvaluator class.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a randomly initialized evaluator once for all tests."""
        # Disable GPU for testing to avoid VRAM issues and ensure consistency
        tf.config.set_visible_devices([], "GPU")
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

        # Test value output
        self.assertIsInstance(value, float, "Value should be a float.")
        self.assertTrue(-3.0 <= value <= 3.0, "Value should be in the [-3, 3] range.")

        # Test policy output
        self.assertIsInstance(policy, np.ndarray, "Policy should be a numpy array.")
        self.assertEqual(
            policy.shape,
            (ACTION_SPACE_SIZE,),
            f"Policy shape should be ({ACTION_SPACE_SIZE},).",
        )
        self.assertAlmostEqual(
            np.sum(policy),
            1.0,
            places=5,
            msg="Sum of policy probabilities should be 1.0.",
        )

    def test_illegal_move_masking(self):
        """The most important test: ensures probabilities for illegal moves are zero."""
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        _, policy = self.evaluator.evaluate(board)

        legal_move_indices = {move_to_index(m) for m in board.legal_moves}

        illegal_move_count = 0
        for i in range(ACTION_SPACE_SIZE):
            if i not in legal_move_indices:
                self.assertEqual(
                    policy[i], 0.0, f"Policy for illegal move index {i} should be 0.0."
                )
                illegal_move_count += 1

        self.assertGreater(
            illegal_move_count,
            ACTION_SPACE_SIZE - len(list(board.legal_moves)) - 50,  # allow some slack
            "A substantial number of illegal moves should have zero probability.",
        )

    def test_black_to_move_evaluation(self):
        """
        Tests that the network gives a consistent evaluation of a position
        regardless of whose turn it is, using the board mirroring technique.
        """
        # A standard position where it's black's turn
        board_b = chess.Board(
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )
        board_b.push_uci("g1f3")

        # Create the exact same position from White's perspective by mirroring
        board_w = board_b.mirror()

        # Sanity checks for the test setup
        self.assertEqual(board_b.turn, chess.BLACK)
        self.assertEqual(board_w.turn, chess.WHITE)

        # Evaluate both positions
        value_b, policy_b = self.evaluator.evaluate(board_b)
        value_w, policy_w = self.evaluator.evaluate(board_w)

        # The values should be nearly identical because the model sees the same position (one is mirrored).
        self.assertAlmostEqual(
            value_b,
            value_w,
            places=5,
            msg=f"Values for pos and its mirror should be almost identical. Got B:{value_b}, W:{value_w}",
        )

        # The policy for black's turn (policy_b) is already un-mirrored by the evaluator.
        # The policy for white's turn (policy_w) corresponds to the mirrored board.
        # To compare them, we must remap policy_w to the original board's perspective.
        remapped_policy_w = unmirror_policy(policy_w)

        self.assertTrue(
            np.allclose(policy_b, remapped_policy_w, atol=1e-5),
            "The un-mirrored black policy should be almost identical to the remapped white policy.",
        )

    def test_terminal_node_evaluation(self):
        """Tests evaluation on a checkmated or stalemate board."""
        # Fool's Mate - a known, simple checkmate.
        board = chess.Board()
        board.push_uci("f2f3")
        board.push_uci("e7e5")
        board.push_uci("g2g4")
        board.push_uci("d8h4")
        self.assertTrue(board.is_checkmate())

        value, policy = self.evaluator.evaluate(board)

        # The value for a lost position should be -1.0
        self.assertAlmostEqual(
            value,
            -1.0,
            places=5,
            msg="Value for a checkmated position (for the mated player) should be -1.0.",
        )

        # The policy for a terminal node should have no legal moves, sum should be 0
        self.assertAlmostEqual(
            np.sum(policy),
            0.0,
            places=5,
            msg="Sum of policy for a terminal node should be 0.0 as there are no legal moves.",
        )


if __name__ == "__main__":
    unittest.main()
