# src/tests/test_move_mapping.py

import unittest
import chess
from src.move_mapping import move_to_index, index_to_move


class TestMoveMapping(unittest.TestCase):
    """
    Tests the move mapping functions for correctness and robustness.
    """

    def test_round_trip_conversion(self):
        """
        Tests that converting a move to an index and back yields the original move.
        Covers standard moves, captures, and castling.
        """
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )

        # A variety of move types to test
        moves_to_test = [
            chess.Move.from_uci("e5d7"),  # Knight capture
            chess.Move.from_uci("f3h3"),  # Queen move
            chess.Move.from_uci("e1g1"),  # Kingside castling
            chess.Move.from_uci("a1b1"),  # Rook move
            chess.Move.from_uci("d2g5"),  # Bishop move
        ]

        for move in moves_to_test:
            with self.subTest(move=move.uci()):
                move_idx = move_to_index(move)
                self.assertIsNotNone(
                    move_idx, f"Move {move.uci()} should have a valid index."
                )

                retrieved_move = index_to_move(move_idx)
                self.assertEqual(
                    move, retrieved_move, f"Round trip for {move.uci()} failed."
                )

    def test_promotion_edge_cases(self):
        """
        Ensures all four promotion types for a single pawn move map to unique indices.
        """
        board = chess.Board("rnbqkbnr/pPpppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")

        promotion_ucis = ["b7a8q", "b7a8r", "b7a8b", "b7a8n"]
        promotion_moves = [chess.Move.from_uci(uci) for uci in promotion_ucis]

        indices = set()
        for move in promotion_moves:
            with self.subTest(move=move.uci()):
                move_idx = move_to_index(move)
                self.assertIsNotNone(
                    move_idx, f"Move {move.uci()} should have a valid index."
                )
                self.assertNotIn(
                    move_idx, indices, f"Index for {move.uci()} should be unique."
                )
                indices.add(move_idx)

    def test_en_passant_edge_case(self):
        """
        Tests round-trip conversion for a specific en passant capture.
        """
        # Set up a position where en passant is possible
        board = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        en_passant_move = chess.Move.from_uci("e5f6")

        self.assertTrue(
            board.is_en_passant(en_passant_move), "e5f6 should be an en passant move."
        )

        move_idx = move_to_index(en_passant_move)
        self.assertIsNotNone(move_idx, "En passant move should have a valid index.")

        retrieved_move = index_to_move(move_idx)
        self.assertEqual(
            en_passant_move, retrieved_move, "Round trip for en passant move failed."
        )

    def test_boundary_checks_with_legal_moves(self):
        """
        From a complex mid-game position, ensures all legal moves can be mapped
        to an index without errors.
        """
        board = chess.Board(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
        )

        for move in board.legal_moves:
            with self.subTest(move=move.uci()):
                move_idx = move_to_index(move)
                self.assertIsNotNone(
                    move_idx, f"Legal move {move.uci()} failed to map to an index."
                )

    def test_castling_representation(self):
        """Ensures castling moves map to the correct surrogate rook-move indices."""
        castling_pairs = [
            ("e1g1", "e1h1"),  # White kingside
            ("e1c1", "e1a1"),  # White queenside
            ("e8g8", "e8h8"),  # Black kingside
            ("e8c8", "e8a8"),  # Black queenside
        ]

        for castle_uci, surrogate_uci in castling_pairs:
            with self.subTest(castle=castle_uci):
                idx_castle = move_to_index(chess.Move.from_uci(castle_uci))
                idx_surrogate = move_to_index(chess.Move.from_uci(surrogate_uci))
                self.assertIsNotNone(
                    idx_castle, f"Index for {castle_uci} should not be None."
                )
                self.assertIsNotNone(
                    idx_surrogate, f"Index for {surrogate_uci} should not be None."
                )
                self.assertEqual(
                    idx_castle,
                    idx_surrogate,
                    f"Castling move {castle_uci} should map to the same index as {surrogate_uci}.",
                )


if __name__ == "__main__":
    unittest.main()
