from multiprocessing import Process, Queue
from typing import Dict, Optional, Protocol
import chess
import numpy as np
import logging
from dataclasses import dataclass
import tensorflow as tf

from src.mcts.node import MCTSNode
from src.encoder import Encoder
from src.move_mapping import index_to_move, ACTION_SPACE_SIZE, move_to_index

logger = logging.getLogger(__name__)


@dataclass
class SearchTask:
    """A dataclass to represent a search task for the worker."""

    fen: str
    num_simulations: int


@dataclass
class SearchResult:
    """A dataclass to represent the result of a search."""

    fen: str
    policy: Dict[chess.Move, float]
    error: Optional[str] = None


class Evaluator(Protocol):
    """
    Defines the interface for a position evaluator.
    This allows swapping different evaluation methods (e.g., mock, remote NN).
    """

    def evaluate(self, board: chess.Board) -> tuple[float, Dict[chess.Move, float]]: ...


class RemoteEvaluator:
    """
    An evaluator that communicates with a remote EvaluationManager process.
    """

    def __init__(self, worker_id: int, request_q: Queue, response_q: Queue):
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self.encoder = Encoder()

    def evaluate(self, board: chess.Board) -> tuple[float, Dict[chess.Move, float]]:
        """
        Encodes the board, sends it for evaluation, and waits for the result.
        """
        encoded_state = self.encoder.encode(board)
        request = {"worker_id": self.worker_id, "encoded_state": encoded_state}
        self.request_q.put(request)

        # *** FIX: Correctly handle responses from the shared queue to prevent deadlock ***
        while True:
            response = self.response_q.get()
            if response.get("worker_id") == self.worker_id:
                # This response is for me.
                if "error" in response:
                    raise RuntimeError(f"Evaluation failed: {response['error']}")

                policy_logits = response["policy_logits"]
                policy = self._decode_policy(policy_logits, board)

                return response["value"], policy
            else:
                # This response is for another worker. Put it back on the queue.
                self.response_q.put(response)

    def _decode_policy(
        self, logits: np.ndarray, board: chess.Board
    ) -> Dict[chess.Move, float]:
        """Converts raw policy logits into a dictionary of legal moves and their probabilities."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        legal_move_indices = [
            move_to_index(m, board)
            for m in legal_moves
            if move_to_index(m, board) is not None
        ]

        if not legal_move_indices:
            return {}

        mask = np.full(logits.shape, -np.inf, dtype=np.float32)
        mask[legal_move_indices] = 0.0
        masked_logits = logits + mask

        # Use TensorFlow for softmax as it's already a dependency and handles edge cases.
        probabilities = tf.nn.softmax(masked_logits).numpy()

        # Reconstruct the policy dictionary for legal moves
        policy_dict = {
            move: float(probabilities[move_to_index(move, board)])
            for move in legal_moves
            if move_to_index(move, board) is not None
        }

        # Normalize to ensure the sum is exactly 1.0 due to potential floating point inaccuracies
        total_prob = sum(policy_dict.values())
        if total_prob > 0:
            return {move: prob / total_prob for move, prob in policy_dict.items()}
        return {}


class MCTS:
    """
    Encapsulates the core Monte Carlo Tree Search algorithm.
    """

    def __init__(self, evaluator: Evaluator, c_puct: float, n_scl: int):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_scl = n_scl

    def run_search(
        self, board: chess.Board, num_simulations: int
    ) -> Dict[chess.Move, float]:
        """
        Performs the MCTS search for a given board state.
        """
        root = MCTSNode(depth=0)

        for _ in range(num_simulations):
            node = root
            search_board = board.copy()

            # 1. Selection
            while not node.is_leaf():
                # *** FIX: Handle the case where select_child returns None ***
                child_data = node.select_child(self.c_puct, self.n_scl)
                if child_data is None:
                    # This node has no children to select, so it's effectively a leaf for this path
                    break
                move, node = child_data
                search_board.push(move)

            # 2. Expansion & Evaluation
            value = 0.0
            if not search_board.is_game_over(claim_draw=True):
                value, policy = self.evaluator.evaluate(search_board)
                node.expand(policy)
            else:
                value = self._get_game_outcome(search_board)

            # 3. Backpropagation
            node.update(value)

        return self._calculate_final_policy(root)

    def _get_game_outcome(self, board: chess.Board) -> float:
        """Determines the game outcome from the perspective of the current player."""
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _calculate_final_policy(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """Calculates policy from visit counts."""
        if root.is_leaf() or root.visit_count == 0:
            return {}

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return {}

        return {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }


class SearchWorker(Process):
    """
    A worker process that manages MCTS tasks.
    """

    def __init__(
        self,
        worker_id: int,
        task_q: Queue,
        result_q: Queue,
        request_q: Queue,
        response_q: Queue,
        c_puct: float = 1.0,
        n_scl: int = 1000,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.task_q = task_q
        self.result_q = result_q
        self.request_q = request_q
        self.response_q = response_q
        self.c_puct = c_puct
        self.n_scl = n_scl

    def run(self):
        """Main loop: get task, run MCTS, put result."""
        evaluator = RemoteEvaluator(self.worker_id, self.request_q, self.response_q)
        mcts_instance = MCTS(evaluator, self.c_puct, self.n_scl)

        while True:
            task: Optional[SearchTask] = self.task_q.get()
            if task is None:
                logger.info(f"Worker {self.worker_id} shutting down.")
                break

            try:
                board = chess.Board(task.fen)
                policy = mcts_instance.run_search(board, task.num_simulations)
                result = SearchResult(fen=task.fen, policy=policy)
                self.result_q.put(result)
            except Exception as e:
                logger.error(
                    f"Error in worker {self.worker_id} for FEN {task.fen}: {e}",
                    exc_info=True,
                )
                result = SearchResult(fen=task.fen, policy={}, error=str(e))
                self.result_q.put(result)
