# src/move_mapping.py

import chess
import logging
from typing import List, Optional

ACTION_SPACE_SIZE = 4672
_QUEEN_MOVE_COUNT = 56
_KNIGHT_MOVE_COUNT = 8
_UNDERPROMOTION_COUNT = 9
_MOVE_TYPES_PER_SQUARE = (
    _QUEEN_MOVE_COUNT + _KNIGHT_MOVE_COUNT + _UNDERPROMOTION_COUNT
)  # 73

_QUEEN_DIRECTIONS = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]
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
_UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_INDEX_TO_MOVE_LIST: Optional[List[Optional[chess.Move]]] = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _get_move_plane_from_board(move: chess.Move, board: chess.Board) -> Optional[int]:
    """Calculates the move plane index (0-72) for a given move, using board context."""
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None

    from_sq, to_sq = move.from_square, move.to_square
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = to_file - from_file, to_rank - from_rank

    if (
        moving_piece.piece_type == chess.PAWN
        and move.promotion in _UNDERPROMOTION_PIECES
    ):
        try:
            piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
            direction_idx = df + 1  # -1 -> 0, 0 -> 1, 1 -> 2
            return (
                _QUEEN_MOVE_COUNT + _KNIGHT_MOVE_COUNT + (piece_idx * 3 + direction_idx)
            )
        except ValueError:
            return None

    if moving_piece.piece_type == chess.KNIGHT:
        try:
            return _QUEEN_MOVE_COUNT + _KNIGHT_OFFSETS.index((df, dr))
        except ValueError:
            return None

    dist = max(abs(df), abs(dr))
    if dist == 0:
        return None
    direction = (df // dist, dr // dist)
    try:
        direction_idx = _QUEEN_DIRECTIONS.index(direction)
        return direction_idx * 7 + (dist - 1)
    except ValueError:
        return None


def _create_move_mappings():
    """Generates the decoder list (index -> move)."""
    global _INDEX_TO_MOVE_LIST
    if _INDEX_TO_MOVE_LIST is not None:
        return

    temp_index_to_move = [None] * ACTION_SPACE_SIZE
    for from_sq in chess.SQUARES:
        # Queen-like moves
        for i in range(_QUEEN_MOVE_COUNT):
            direction_idx, dist = i // 7, (i % 7) + 1
            df, dr = _QUEEN_DIRECTIONS[direction_idx]
            to_file, to_rank = (
                chess.square_file(from_sq) + df * dist,
                chess.square_rank(from_sq) + dr * dist,
            )
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_sq = chess.square(to_file, to_rank)
                index = from_sq * _MOVE_TYPES_PER_SQUARE + i
                # *** FIX: Do NOT create promotion moves here. A queen promotion move like
                # e7e8q is represented by the same plane as a queen move e7e8.
                temp_index_to_move[index] = chess.Move(from_sq, to_sq)

        # Knight moves
        for i in range(_KNIGHT_MOVE_COUNT):
            df, dr = _KNIGHT_OFFSETS[i]
            to_file, to_rank = (
                chess.square_file(from_sq) + df,
                chess.square_rank(from_sq) + dr,
            )
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                index = from_sq * _MOVE_TYPES_PER_SQUARE + _QUEEN_MOVE_COUNT + i
                temp_index_to_move[index] = chess.Move(
                    from_sq, chess.square(to_file, to_rank)
                )

        # Underpromotions
        for i in range(_UNDERPROMOTION_COUNT):
            promo_piece = _UNDERPROMOTION_PIECES[i // 3]
            df = (i % 3) - 1
            for rank_info in [(6, 7), (1, 0)]:  # White, Black
                if chess.square_rank(from_sq) == rank_info[0]:
                    to_file = chess.square_file(from_sq) + df
                    if 0 <= to_file < 8:
                        to_sq = chess.square(to_file, rank_info[1])
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
    log.info("Action space decoder successfully generated.")


def move_to_index(move: chess.Move, board: chess.Board) -> Optional[int]:
    """Converts a chess.Move object to its corresponding unique integer index."""
    plane = _get_move_plane_from_board(move, board)
    if plane is None:
        return None
    return move.from_square * _MOVE_TYPES_PER_SQUARE + plane


def index_to_move(index: int) -> Optional[chess.Move]:
    """Converts an integer index back to a potential chess.Move object."""
    if _INDEX_TO_MOVE_LIST is None:
        _create_move_mappings()
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise IndexError(f"Move index {index} is out of bounds.")
    return _INDEX_TO_MOVE_LIST[index]


_create_move_mappings()
