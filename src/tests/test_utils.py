# src/tests/test_utils.py

import unittest
import chess
import numpy as np
from src.utils import unmirror_policy, UNMIRROR_MAP
from src.move_mapping import move_to_index, index_to_move, ACTION_SPACE_SIZE

class TestUtils(unittest.TestCase):
    """
    Tests for utility functions like policy mirroring.
    """

    def test_unmirror_policy_simple_move(self):
        """
        Tests that the unmirror_policy function correctly remaps a policy
        vector from a mirrored board to the original orientation for a simple move.
        """
        policy = np.zeros(ACTION_SPACE_SIZE)
        unique_value = 0.99

        # A standard opening move for White and its mirrored equivalent for Black.
        board_white = chess.Board()
        move_white = chess.Move.from_uci("e2e4")

        board_black = chess.Board()
        board_black.push(move_white) # Make a move to get to black's turn
        move_black = chess.Move.from_uci("e7e5")

        # The model sees e2e4 when Black plays e7e5 on a mirrored board.
        # We want to ensure the probability at e2e4 gets remapped to e7e5.
        idx_e2e4 = move_to_index(move_white, board_white)
        idx_e7e5 = move_to_index(move_black, board_black)

        self.assertIsNotNone(idx_e2e4, "Index for e2e4 should be valid.")
        self.assertIsNotNone(idx_e7e5, "Index for e7e5 should be valid.")

        # Set the probability on the "mirrored" move index
        policy[idx_e2e4] = unique_value

        unmirrored_policy = unmirror_policy(policy)

        # Assert that the value has moved to the correct un-mirrored index
        self.assertAlmostEqual(unmirrored_policy[idx_e7e5], unique_value, places=5,
                             msg="The probability for e2e4 should move to e7e5 after un-mirroring.")
        self.assertAlmostEqual(unmirrored_policy[idx_e2e4], 0.0, places=5,
                             msg="The original e2e4 index should be empty after un-mirroring.")

    def test_unmirror_involution(self):
        """
        Applying the UNMIRROR_MAP twice should return the identity mapping.
        This is a critical property for correctness.
        """
        # Apply the mapping twice
        remapped_map = UNMIRROR_MAP[UNMIRROR_MAP]
        # The identity mapping is just an array [0, 1, 2, ..., N-1]
        identity = np.arange(ACTION_SPACE_SIZE)

        self.assertTrue(np.array_equal(remapped_map, identity),
                        "UNMIRROR_MAP should be an involution (mapping applied twice yields identity).")


if __name__ == "__main__":
    unittest.main()
