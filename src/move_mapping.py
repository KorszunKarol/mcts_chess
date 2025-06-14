# src/move_mapping.py

import chess
import logging
from typing import List, Optional

# This module implements the canonical AlphaZero-style move representation,
# which encodes all possible moves into a flat vector of size 4672.
# The encoding is based on representing moves as a 'from' square combined
# with a 'move type' plane. There are 64 possible 'from' squares and 73
# possible move types from each square (64 * 73 = 4672).

ACTION_SPACE_SIZE = 4672

# --- Move Plane Configuration ---
# 73 total move planes per square:
# - 56 for queen-like moves (rooks, bishops, queens)
# - 8 for knight moves
# - 9 for pawn underpromotions (N, B, R promotions in 3 directions)
_QUEEN_MOVE_COUNT = 56
_KNIGHT_MOVE_COUNT = 8
_UNDERPROMOTION_COUNT = 9
_MOVE_TYPES_PER_SQUARE = (
    _QUEEN_MOVE_COUNT + _KNIGHT_MOVE_COUNT + _UNDERPROMOTION_COUNT
)  # 73

# --- Pre-computed data for mapping logic ---
_QUEEN_DIRECTIONS = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]  # N, NE, E, SE, S, SW, W, NW
_KNIGHT_OFFSETS = [
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1),
]
_UNDERPROMOTION_PIECES = [
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
]  # Note: Order matters for indexing

# --- Caches to store the generated mappings ---
_INDEX_TO_MOVE_LIST: Optional[List[Optional[chess.Move]]] = None

# --- Logger ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _get_move_plane(move: chess.Move) -> Optional[int]:
    """Calculates the move plane index (0-72) for a given move."""
    from_sq, to_sq = move.from_square, move.to_square
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = to_file - from_file, to_rank - from_rank

    # Handle underpromotions first as they are specific
    if move.promotion and move.promotion in _UNDERPROMOTION_PIECES:
        try:
            piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
            # df can be -1 (left capture), 0 (straight), 1 (right capture)
            direction_idx = df + 1
            return (
                _QUEEN_MOVE_COUNT + _KNIGHT_MOVE_COUNT + (piece_idx * 3 + direction_idx)
            )
        except ValueError:
            return None

    # Handle knight moves
    try:
        # Check if the offset matches a knight move
        knight_move_idx = _KNIGHT_OFFSETS.index((df, dr))
        return _QUEEN_MOVE_COUNT + knight_move_idx
    except ValueError:
        pass  # Not a knight move, proceed to check queen moves

    # Handle queen-like moves (includes queen promotions)
    dist = max(abs(df), abs(dr))
    if dist == 0:
        return None

    direction = (df // dist, dr // dist)
    try:
        direction_idx = _QUEEN_DIRECTIONS.index(direction)
        # 7 possible distances for each of the 8 directions
        return direction_idx * 7 + (dist - 1)
    except ValueError:
        return None  # Not a queen-like move


def _create_move_mappings():
    """
    Generates and populates the move mapping list. This function acts as the
    'decoder' for the action space, mapping an index to a potential move.
    """
    global _INDEX_TO_MOVE_LIST
    if _INDEX_TO_MOVE_LIST is not None:
        return

    temp_index_to_move = [None] * ACTION_SPACE_SIZE
    for from_sq in chess.SQUARES:
        # 1. Queen moves
        for i in range(_QUEEN_MOVE_COUNT):
            direction_idx = i // 7
            dist = (i % 7) + 1
            df, dr = _QUEEN_DIRECTIONS[direction_idx]
            to_file = chess.square_file(from_sq) + df * dist
            to_rank = chess.square_rank(from_sq) + dr * dist
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_sq = chess.square(to_file, to_rank)
                index = from_sq * _MOVE_TYPES_PER_SQUARE + i
                # Handle queen promotions
                if (chess.square_rank(from_sq) in {1, 6}) and (abs(df) <= 1):
                    if (chess.square_rank(from_sq) == 6 and to_rank == 7) or (
                        chess.square_rank(from_sq) == 1 and to_rank == 0
                    ):
                        temp_index_to_move[index] = chess.Move(
                            from_sq, to_sq, promotion=chess.QUEEN
                        )
                        continue
                temp_index_to_move[index] = chess.Move(from_sq, to_sq)

        # 2. Knight moves
        for i in range(_KNIGHT_MOVE_COUNT):
            df, dr = _KNIGHT_OFFSETS[i]
            to_file = chess.square_file(from_sq) + df
            to_rank = chess.square_rank(from_sq) + dr
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_sq = chess.square(to_file, to_rank)
                index = from_sq * _MOVE_TYPES_PER_SQUARE + _QUEEN_MOVE_COUNT + i
                temp_index_to_move[index] = chess.Move(from_sq, to_sq)

        # 3. Underpromotions
        for i in range(_UNDERPROMOTION_COUNT):
            promo_piece_idx = i // 3
            direction_idx = (i % 3) - 1  # -1, 0, 1
            promo_piece = _UNDERPROMOTION_PIECES[promo_piece_idx]
            # Assume white promotion from 7th rank
            if chess.square_rank(from_sq) == 6:
                to_file = chess.square_file(from_sq) + direction_idx
                if 0 <= to_file < 8:
                    to_sq = chess.square(to_file, 7)
                    index = (
                        from_sq * _MOVE_TYPES_PER_SQUARE
                        + _QUEEN_MOVE_COUNT
                        + _KNIGHT_MOVE_COUNT
                        + i
                    )
                    temp_index_to_move[index] = chess.Move(
                        from_sq, to_sq, promotion=promo_piece
                    )
            # Assume black promotion from 2nd rank
            if chess.square_rank(from_sq) == 1:
                to_file = chess.square_file(from_sq) + direction_idx
                if 0 <= to_file < 8:
                    to_sq = chess.square(to_file, 0)
                    index = (
                        from_sq * _MOVE_TYPES_PER_SQUARE
                        + _QUEEN_MOVE_COUNT
                        + _KNIGHT_MOVE_COUNT
                        + i
                    )
                    temp_index_to_move[index] = chess.Move(
                        from_sq, to_sq, promotion=promo_piece
                    )

    _INDEX_TO_MOVE_LIST = temp_index_to_move
    log.info(
        f"Action space decoder successfully generated. Total potential moves: {ACTION_SPACE_SIZE}."
    )


def move_to_index(move: chess.Move) -> Optional[int]:
    """
    Converts a chess.Move object to its corresponding unique integer index
    by calculating its 'from' square and its 'move plane'.
    """
    # First, handle the special case for castling by using the surrogate rook move
    # for the index calculation. The actual move object passed to the board
    # will still be the correct castling move.
    uci = move.uci()
    if uci == "e1g1":
        move_for_indexing = chess.Move.from_uci("e1h1")
    elif uci == "e1c1":
        move_for_indexing = chess.Move.from_uci("e1a1")
    elif uci == "e8g8":
        move_for_indexing = chess.Move.from_uci("e8h8")
    elif uci == "e8c8":
        move_for_indexing = chess.Move.from_uci("e8a8")
    else:
        move_for_indexing = move

    plane = _get_move_plane(move_for_indexing)

    if plane is None:
        # It's useful to know the original move when logging a warning.
        log.warning(
            f"Could not determine move plane for {move_for_indexing.uci()} (from original move {uci})"
        )
        return None

    index = move_for_indexing.from_square * _MOVE_TYPES_PER_SQUARE + plane

    if 0 <= index < ACTION_SPACE_SIZE:
        return index

    log.error(
        f"Calculated index {index} for move {move_for_indexing.uci()} is out of bounds."
    )
    return None


def index_to_move(index: int) -> Optional[chess.Move]:
    if _INDEX_TO_MOVE_LIST is None:
        _create_move_mappings()
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise IndexError(
            f"Move index {index} is out of bounds for action space size {ACTION_SPACE_SIZE}."
        )
    return _INDEX_TO_MOVE_LIST[index]


_create_move_mappings()
