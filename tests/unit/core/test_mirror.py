import chess
import numpy as np

from src.move_mapping import move_to_index
from src.utils import UNMIRROR_MAP


def test_unmirror_map_e7e5_to_e2e4():
    board = chess.Board()
    idx_e2e4 = move_to_index(chess.Move.from_uci("e2e4"), board)
    idx_e7e5 = move_to_index(chess.Move.from_uci("e7e5"), board)

    assert idx_e2e4 is not None
    assert idx_e7e5 is not None

    mapped_idx = UNMIRROR_MAP[idx_e7e5]
    assert mapped_idx == idx_e2e4


def test_unmirror_logits_value_propagation():
    board = chess.Board()
    idx_e2e4 = move_to_index(chess.Move.from_uci("e2e4"), board)
    idx_e7e5 = move_to_index(chess.Move.from_uci("e7e5"), board)
    assert idx_e2e4 is not None and idx_e7e5 is not None

    logits = np.zeros(4672)
    logits[idx_e2e4] = 100.0

    unmirrored_logits = logits[UNMIRROR_MAP]
    assert unmirrored_logits[idx_e7e5] == 100.0

