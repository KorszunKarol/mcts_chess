# src/tests/test_move_mapping.py

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
        # *** FIX: Corrected the invalid FEN string ***
        board = chess.Board("rnbqk2r/pPpp1ppp/5n2/8/8/8/1P1PPPPP/RNBQKBNR w KQkq - 0 1")

        # Test a promotion by capture
        promotion_ucis = ["b7a8q", "b7a8r", "b7a8b", "b7a8n"]
        promotion_moves = [chess.Move.from_uci(uci) for uci in promotion_ucis]
        indices = set()

        for move in promotion_moves:
            with self.subTest(move=move.uci()):
                move_idx = move_to_index(move, board)
                self.assertIsNotNone(
                    move_idx, f"Promotion move {move.uci()} should have a valid index."
                )
                self.assertNotIn(
                    move_idx, indices, f"Index for {move.uci()} should be unique."
                )
                indices.add(move_idx)

    def test_full_action_space_round_trip(self):
        """
        Iterates the entire action space, decodes the index to a move, creates
        a board context where that move is plausible, and verifies that encoding
        it again yields the original index.
        """
        for index in range(ACTION_SPACE_SIZE):
            move = index_to_move(index)
            if move is None:
                continue

            board = chess.Board(fen=None)
            color = (
                chess.WHITE if chess.square_rank(move.from_square) < 4 else chess.BLACK
            )
            piece = self._get_piece_for_move(move, color)
            board.set_piece_at(move.from_square, piece)
            board.turn = color

            retrieved_index = move_to_index(move, board)

            self.assertEqual(
                index,
                retrieved_index,
                f"Round trip failed for index {index} (Move: {move.uci()}). Got back {retrieved_index}.",
            )

    def _get_piece_for_move(self, move: chess.Move, color: chess.Color) -> chess.Piece:
        """Helper to determine what piece could be making a given move."""
        if move.promotion:
            return chess.Piece(chess.PAWN, color)

        df = abs(
            chess.square_file(move.from_square) - chess.square_file(move.to_square)
        )
        dr = abs(
            chess.square_rank(move.from_square) - chess.square_rank(move.to_square)
        )

        if (df == 1 and dr == 2) or (df == 2 and dr == 1):
            return chess.Piece(chess.KNIGHT, color)

        if df == 0 or dr == 0 or df == dr:
            if dr == 1 and df == 0 and not move.promotion:
                # Check if it's a pawn double-push
                if (
                    color == chess.WHITE and chess.square_rank(move.from_square) == 1
                ) or (
                    color == chess.BLACK and chess.square_rank(move.from_square) == 6
                ):
                    return chess.Piece(chess.PAWN, color)
                return chess.Piece(chess.KING, color)  # Assume king for single-step
            if dr <= 1 and df <= 1:
                return chess.Piece(chess.KING, color)
            return chess.Piece(chess.QUEEN, color)

        return chess.Piece(chess.QUEEN, color)  # Fallback


if __name__ == "__main__":
    unittest.main()
