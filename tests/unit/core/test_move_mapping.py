# tests/unit/core/test_move_mapping.py

import unittest
import chess
from src.move_mapping import move_to_index, index_to_move, ACTION_SPACE_SIZE


class TestMoveMapping(unittest.TestCase):
    """
    Tests the move mapping functions for correctness and robustness, ensuring
    the board context is used correctly.
    """

    def test_round_trip_conversion(self):
        """
        Tests that converting a move to an index and back yields the original move
        from a complex middlegame position.
        """
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )

        for move in board.legal_moves:
            with self.subTest(move=move.uci()):
                move_idx = move_to_index(move, board)
                self.assertIsNotNone(
                    move_idx, f"Move {move.uci()} should have a valid index."
                )

                retrieved_move = index_to_move(move_idx)
                self.assertIsNotNone(
                    retrieved_move, f"Index {move_idx} should decode to a valid move."
                )

                self.assertEqual(
                    move.uci(),
                    retrieved_move.uci(),
                    f"Round trip for {move.uci()} failed.",
                )

    def test_promotion_uniqueness(self):
        """
        Ensures all four promotion types for a single pawn move map to unique indices.
        """
        board = chess.Board("rnbqk2r/pPpp1ppp/5n2/8/8/8/1P1PPPPP/RNBQKBNR w KQkq - 0 1")

        # Test a promotion by capture
        promotion_ucis = ["b7a8q", "b7a8r", "b7a8b", "b7a8n"]
        promotion_moves = [chess.Move.from_uci(uci) for uci in promotion_ucis]
        indices = set()

        for move in promotion_moves:
            with self.subTest(move=move.uci()):
                idx = move_to_index(move, board)
                self.assertIsNotNone(idx, f"Promotion move {move} should have an index.")
                self.assertNotIn(idx, indices, "Promotion indices should be unique.")
                indices.add(idx)

    def test_index_range(self):
        """Ensure indices stay within action space size."""
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            self.assertLess(idx, ACTION_SPACE_SIZE)


if __name__ == "__main__":
    unittest.main()

