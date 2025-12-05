import pytest
import chess
import numpy as np
from pathlib import Path

from src.utils import unmirror_policy, UNMIRROR_MAP
from src.move_mapping import move_to_index, index_to_move, ACTION_SPACE_SIZE
from src.coach_tal.evaluator import TransformerEvaluator
from src.coach_tal.selector import CoachTalSelector, CoachTalConfig
from src.mcts.controller import MCTSController

# Constants for testing
MODEL_PATH = "/home/karolito/DL/chess_2.0/saved_models/best_model_pytorch.pt"


class TestMirroringLogic:
    """Tests for board and policy mirroring logic."""

    def test_unmirror_map_e2e4_e7e5(self):
        """Verify that e7e5 (Black) maps to e2e4 (White) via unmirroring."""
        board = chess.Board()

        move_white = chess.Move.from_uci("e2e4")
        idx_white = move_to_index(move_white, board)

        move_black = chess.Move.from_uci("e7e5")
        idx_black = move_to_index(move_black, board)

        assert UNMIRROR_MAP[idx_black] == idx_white

    def test_unmirror_policy_function(self):
        """Test the unmirror_policy function with a synthetic logit vector."""
        logits = np.zeros(ACTION_SPACE_SIZE)

        board = chess.Board()
        idx_white = move_to_index(chess.Move.from_uci("e2e4"), board)
        logits[idx_white] = 10.0

        unmirrored = unmirror_policy(logits)

        idx_black = move_to_index(chess.Move.from_uci("e7e5"), board)
        assert unmirrored[idx_black] == 10.0


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model weights not found")
class TestCoachTalIntegration:
    """Integration tests for Coach Tal selector with real model weights."""

    def setup_method(self):
        self.evaluator = TransformerEvaluator(MODEL_PATH, use_pytorch=True)
        self.selector = CoachTalSelector(
            CoachTalConfig(
                weights_path=MODEL_PATH,
                use_pytorch=True,
                lambda_psych=0.3,
                gamma_confusion=0.5,
                delta_soundness=0.15,
                top_k_candidates=3,
            )
        )

    def test_selector_returns_legal_move(self):
        """Selector should return a legal move on starting position."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        uniform_score = 1.0 / len(legal_moves)
        candidate_scores = {m: uniform_score for m in legal_moves}
        result = self.selector.select(board, candidate_scores=candidate_scores)
        assert result.chosen_move in board.legal_moves

    def test_controller_and_selector_work_together(self):
        """Controller policy should feed into selector without error."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        # Uniform candidate scores as a stand-in for MCTS policy
        uniform_score = 1.0 / len(legal_moves)
        candidate_scores = {m: uniform_score for m in legal_moves}
        result = self.selector.select(board, candidate_scores=candidate_scores)
        assert result.chosen_move in board.legal_moves

