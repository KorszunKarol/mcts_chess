# src/tests/test_utils.py
import unittest
import chess
import numpy as np
from src.utils import unmirror_policy, UNMIRROR_MAP
from src.move_mapping import move_to_index, ACTION_SPACE_SIZE


class TestUtils(unittest.TestCase):
    """
    Tests for utility functions like policy mirroring.
    """

    def test_unmirror_policy(self):
        """
        Tests that the unmirror_policy function correctly remaps a policy
        vector from a mirrored board to the original orientation.
        """
        # 1. Create a sample policy vector with a unique value at a known index.
        policy = np.zeros(ACTION_SPACE_SIZE)
        unique_value = 0.99

        # 2. Pick a move and its mirrored counterpart.
        # e2e4 is a standard opening move for White.
        # Its mirrored equivalent is e7e5 for Black.
        move_e2e4 = chess.Move.from_uci("e2e4")
        move_e7e5 = chess.Move.from_uci("e7e5")

        # In the unmirroring logic, the *input* policy is from the White POV
        # on a mirrored board. So, if the original move was e7e5 (Black),
        # the model would have seen e2e4 (White). We want to ensure that the
        # probability assigned to e2e4 gets correctly remapped to e7e5.

        idx_e2e4 = move_to_index(move_e2e4)
        idx_e7e5 = move_to_index(move_e7e5)

        self.assertIsNotNone(idx_e2e4, "Index for e2e4 should be valid.")
        self.assertIsNotNone(idx_e7e5, "Index for e7e5 should be valid.")
        self.assertNotEqual(idx_e2e4, idx_e7e5, "Mirrored move indices should differ.")

        policy[idx_e2e4] = unique_value

        # 3. Call the function to un-mirror the policy.
        unmirrored_policy_vec = unmirror_policy(policy)

        # 4. Assert that the unique value has moved to the correct new index.
        self.assertAlmostEqual(
            unmirrored_policy_vec[idx_e7e5],
            unique_value,
            places=5,
            msg="The probability for e2e4 should move to the e7e5 index after un-mirroring.",
        )

        # 5. Assert that the original index is now zero (or close to it).
        self.assertAlmostEqual(
            unmirrored_policy_vec[idx_e2e4],
            0.0,
            places=5,
            msg="The original e2e4 index should be empty after un-mirroring.",
        )

    def test_unmirror_involution(self):
        """Applying the UNMIRROR_MAP twice should return the identity mapping."""
        remapped_map = UNMIRROR_MAP[UNMIRROR_MAP]
        identity = np.arange(ACTION_SPACE_SIZE)
        self.assertTrue(
            np.array_equal(remapped_map, identity),
            "UNMIRROR_MAP should be an involution (mapping applied twice yields identity).",
        )


if __name__ == "__main__":
    unittest.main()
