# src/utils.py

import chess
import numpy as np
from src.move_mapping import ACTION_SPACE_SIZE, move_to_index, index_to_move


def _create_unmirror_map() -> np.ndarray:
    """
    Creates a pre-computed map to translate policy indices from a mirrored
    board back to the original board's perspective.

    Returns:
        A NumPy array where `unmirror_map[i]` is the index corresponding to
        the mirrored version of the move at index `i`.
    """
    unmirror_map = np.arange(ACTION_SPACE_SIZE)
    for i in range(ACTION_SPACE_SIZE):
        move = index_to_move(i)
        if move:
            # Create a board context where this move could be pseudo-legal
            # to correctly determine piece type for re-indexing.
            board = chess.Board(fen=None)
            piece_type = chess.QUEEN # Assume queen by default for geometry
            # A simple heuristic to find the piece type based on move geometry
            df = abs(chess.square_file(move.from_square) - chess.square_file(move.to_square))
            dr = abs(chess.square_rank(move.from_square) - chess.square_rank(move.to_square))
            if (df == 1 and dr == 2) or (df == 2 and dr == 1):
                piece_type = chess.KNIGHT
            if move.promotion:
                piece_type = chess.PAWN

            # Place the piece to create a valid board for move_to_index
            board.set_piece_at(move.from_square, chess.Piece(piece_type, chess.WHITE))

            mirrored_move = chess.Move(
                from_square=chess.square_mirror(move.from_square),
                to_square=chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )

            # The mirrored board context is also needed
            mirrored_board = board.mirror()

            mirrored_index = move_to_index(mirrored_move, mirrored_board)
            if mirrored_index is not None:
                unmirror_map[i] = mirrored_index
    return unmirror_map


# Pre-compute the map when the module is loaded.
UNMIRROR_MAP = _create_unmirror_map()


def unmirror_policy(policy_logits: np.ndarray) -> np.ndarray:
    """
    Remaps a policy vector from a mirrored board to the original orientation.

    Args:
        policy_logits: A NumPy array of shape (ACTION_SPACE_SIZE,) from the
                       model, corresponding to a mirrored board state.

    Returns:
        A NumPy array of the same shape with probabilities remapped to the
        non-mirrored perspective.
    """
    # The remapping is a simple and fast lookup using the pre-computed map.
    return policy_logits[UNMIRROR_MAP]

