from dataclasses import dataclass
import chess
import numpy as np
from typing import Tuple
from src.feature_engineering import (
    get_piece_mobility,
    analyze_pawn_structure,
    center_control,
    piece_square_tables,
    defended_and_vulnerable,
    piece_coordination,
    game_phase,
    king_safety,
)


@dataclass
class Encoder:
    """
    Encodes chess board positions into neural network input tensors.
    Converts chess board states into a (8, 8, 34) tensor representation.
    """

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encodes a chess board into a (8, 8, 34) tensor.

        Channel mapping:
        0-11: Piece positions (p, n, b, r, q, k, P, N, B, R, Q, K)
        12-15: Castling rights
        16: Material score
        17: En passant
        18: Halfmove clock
        19: Fullmove number
        20: Piece mobility
        21-23: Pawn structure (doubled, isolated, passed)
        24: Center control
        25: Piece square tables
        26-29: Defended and vulnerable pieces
        30-31: Piece coordination
        32: Game phase
        33: King safety

        Args:
            board (chess.Board): The chess board to encode.

        Returns:
            np.ndarray: An 8x8x34 tensor representing the board state.
        """
        matrix = np.zeros((8, 8, 34), dtype=np.float32)

        # Basic board representation (channels 0-11)
        self._encode_pieces(board, matrix)

        # Game state features (channels 12-19)
        self._encode_game_state(board, matrix)

        # Advanced features (channels 20-33)
        self._encode_advanced_features(board, matrix)

        # Flip the matrix to maintain consistent orientation
        matrix = np.flip(matrix, axis=0)
        return matrix

    def _encode_pieces(self, board: chess.Board, matrix: np.ndarray):
        """Encodes piece positions into the first 12 channels."""
        piece_to_index = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11,
        }

        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                row, col = divmod(i, 8)
                matrix[row, col, piece_to_index[piece.symbol()]] = 1

    def _encode_game_state(self, board: chess.Board, matrix: np.ndarray):
        """Encodes game state information (channels 12-19)."""
        # Castling rights
        matrix[:, :, 12] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        matrix[:, :, 13] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        matrix[:, :, 14] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        matrix[:, :, 15] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

        # Material score
        matrix[:, :, 16] = self._calculate_material_score(board)

        # En passant
        ep_square = board.ep_square
        if ep_square is not None:
            row, col = divmod(ep_square, 8)
            matrix[row, col, 17] = 1

        # Move counters
        matrix[:, :, 18] = board.halfmove_clock / 100
        matrix[:, :, 19] = board.fullmove_number / 200

    def _encode_advanced_features(self, board: chess.Board, matrix: np.ndarray):
        """Encodes advanced positional features (channels 20-33)."""
        # Mobility
        matrix[:, :, 20] = get_piece_mobility(board)

        # Pawn structure
        doubled, isolated, passed = analyze_pawn_structure(board)
        matrix[:, :, 21] = doubled
        matrix[:, :, 22] = isolated
        matrix[:, :, 23] = passed

        # Center control and piece square tables
        matrix[:, :, 24] = center_control(board)
        matrix[:, :, 25] = piece_square_tables(board)

        # Defended and vulnerable pieces
        defended, vulnerable = defended_and_vulnerable(board)
        matrix[:, :, 26:28] = defended
        matrix[:, :, 28:30] = vulnerable

        # Piece coordination
        coord = piece_coordination(board)
        matrix[:, :, 30:32] = coord

        # Game phase
        matrix[:, :, 32] = game_phase(board)

        # King safety
        king_safety_matrix = king_safety(board)
        matrix[:, :, 33] = king_safety_matrix[:, :, 0]

    @staticmethod
    def _calculate_material_score(board: chess.Board) -> float:
        """Calculates the normalized material score."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        score = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in piece_values.items()
        )
        score -= sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in piece_values.items()
        )
        return score / 39  # Normalize by maximum possible material difference
