# tests/unit/core/test_utils.py

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
        Ensure unmirror_policy remaps probability mass from mirrored to original orientation.
        """
        policy = np.zeros(ACTION_SPACE_SIZE)
        unique_value = 0.99

        board_white = chess.Board()
        move_white = chess.Move.from_uci("e2e4")

        board_black = chess.Board()
        board_black.push(move_white)
        move_black = chess.Move.from_uci("e7e5")

        idx_e2e4 = move_to_index(move_white, board_white)
        idx_e7e5 = move_to_index(move_black, board_black)

        self.assertIsNotNone(idx_e2e4, "Index for e2e4 should be valid.")
        self.assertIsNotNone(idx_e7e5, "Index for e7e5 should be valid.")

        policy[idx_e2e4] = unique_value
        unmirrored_policy = unmirror_policy(policy)

        self.assertAlmostEqual(
            unmirrored_policy[idx_e7e5],
            unique_value,
            places=5,
            msg="The probability for e2e4 should move to e7e5 after un-mirroring.",
        )
        self.assertAlmostEqual(
            unmirrored_policy[idx_e2e4],
            0.0,
            places=5,
            msg="The original e2e4 index should be empty after un-mirroring.",
        )

    def test_unmirror_involution(self):
        """
        Applying the UNMIRROR_MAP twice should return the identity mapping.
        """
        remapped_map = UNMIRROR_MAP[UNMIRROR_MAP]
        identity = np.arange(ACTION_SPACE_SIZE)

        self.assertTrue(
            np.array_equal(remapped_map, identity),
            "UNMIRROR_MAP should be an involution (mapping applied twice yields identity).",
        )


if __name__ == "__main__":
    unittest.main()

