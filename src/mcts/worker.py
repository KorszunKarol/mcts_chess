from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Protocol, TYPE_CHECKING
import chess
import numpy as np
import logging
from dataclasses import dataclass
import tensorflow as tf
import threading
import time

from src.mcts.node import MCTSNode
from src.encoder import Encoder
from src.move_mapping import index_to_move, ACTION_SPACE_SIZE, move_to_index

if TYPE_CHECKING:
    from src.mcts.controller import SharedMemoryConfig

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
    Uses shared memory for high-throughput data transfer.
    """

    def __init__(
        self,
        worker_id: int,
        request_q: Queue,
        response_q: Queue,
        shared_memory_config: Optional["SharedMemoryConfig"] = None,
    ):
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self.encoder = Encoder()
        self.shared_memory_config = shared_memory_config

        self.shared_memory_blocks: Dict[str, SharedMemory] = {}
        self.input_arrays: Dict[str, np.ndarray] = {}
        self.output_arrays: Dict[str, np.ndarray] = {}
        self.buffer_lock = threading.Lock()
        self.next_buffer_index = 0

        if self.shared_memory_config:
            self._setup_shared_memory()

    def _setup_shared_memory(self):
        """Attach to existing shared memory blocks created by the controller."""
        logger.info(
            f"Worker {self.worker_id}: Attaching to {len(self.shared_memory_config.buffer_names)} shared memory blocks..."
        )

        for buffer_name in self.shared_memory_config.buffer_names:
            try:

                shm = SharedMemory(name=buffer_name)
                self.shared_memory_blocks[buffer_name] = shm

                input_size = self.shared_memory_config.get_input_size()
                output_size = self.shared_memory_config.get_output_size()

                input_view = np.frombuffer(
                    shm.buf[:input_size], dtype=self.shared_memory_config.input_dtype
                ).reshape(self.shared_memory_config.input_shape)
                self.input_arrays[buffer_name] = input_view

                output_view = np.frombuffer(
                    shm.buf[input_size : input_size + output_size],
                    dtype=self.shared_memory_config.output_dtype,
                )
                self.output_arrays[buffer_name] = output_view

            except Exception as e:
                logger.error(
                    f"Worker {self.worker_id}: Failed to attach to shared memory {buffer_name}: {e}"
                )
                raise RuntimeError(f"Shared memory setup failed: {e}")

        logger.info(
            f"Worker {self.worker_id}: Successfully attached to {len(self.shared_memory_blocks)} shared memory blocks"
        )

    def _allocate_buffer(self) -> int:
        """
        Allocate a buffer for this evaluation request.
        Uses a simple round-robin allocation strategy.
        """
        with self.buffer_lock:
            buffer_index = self.next_buffer_index % len(
                self.shared_memory_config.buffer_names
            )
            self.next_buffer_index += 1
            return buffer_index

    def _write_input_to_buffer(self, buffer_index: int, encoded_state: np.ndarray):
        """Write input data to a shared memory buffer."""
        buffer_name = self.shared_memory_config.buffer_names[buffer_index]
        input_array = self.input_arrays[buffer_name]
        input_array[:] = encoded_state

    def _read_output_from_buffer(self, buffer_index: int) -> tuple[float, np.ndarray]:
        """Read output data from a shared memory buffer."""
        buffer_name = self.shared_memory_config.buffer_names[buffer_index]
        output_array = self.output_arrays[buffer_name]
        value = float(output_array[0])
        policy_logits = output_array[1:].copy()
        return value, policy_logits

    def evaluate(self, board: chess.Board) -> tuple[float, Dict[chess.Move, float]]:
        """
        Encodes the board, sends it for evaluation, and waits for the result.
        Uses shared memory for high-throughput data transfer when available.
        """
        if self.shared_memory_config:
            return self._evaluate_with_shared_memory(board)
        else:
            return self._evaluate_with_queues(board)

    def _evaluate_with_shared_memory(
        self, board: chess.Board
    ) -> tuple[float, Dict[chess.Move, float]]:
        """Evaluate using shared memory for data transfer."""

        encoded_state = self.encoder.encode(board)

        buffer_index = self._allocate_buffer()

        self._write_input_to_buffer(buffer_index, encoded_state)

        request = {"worker_id": self.worker_id, "buffer_index": buffer_index}
        self.request_q.put(request)

        while True:
            response = self.response_q.get()
            if response.get("worker_id") == self.worker_id:

                if "error" in response:
                    raise RuntimeError(f"Evaluation failed: {response['error']}")

                value, policy_logits = self._read_output_from_buffer(
                    response["buffer_index"]
                )
                policy = self._decode_policy(policy_logits, board)

                return value, policy
            else:

                self.response_q.put(response)

    def _evaluate_with_queues(
        self, board: chess.Board
    ) -> tuple[float, Dict[chess.Move, float]]:
        """Fallback evaluation using queues for backward compatibility."""
        encoded_state = self.encoder.encode(board)
        request = {"worker_id": self.worker_id, "encoded_state": encoded_state}
        self.request_q.put(request)

        while True:
            response = self.response_q.get()
            if response.get("worker_id") == self.worker_id:

                if "error" in response:
                    raise RuntimeError(f"Evaluation failed: {response['error']}")

                policy_logits = response["policy_logits"]
                policy = self._decode_policy(policy_logits, board)

                return response["value"], policy
            else:

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

        probabilities = tf.nn.softmax(masked_logits).numpy()

        policy_dict = {
            move: float(probabilities[move_to_index(move, board)])
            for move in legal_moves
            if move_to_index(move, board) is not None
        }

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

            while not node.is_leaf():

                child_data = node.select_child(self.c_puct, self.n_scl)
                if child_data is None:

                    break
                move, node = child_data
                search_board.push(move)

            value = 0.0
            if not search_board.is_game_over(claim_draw=True):
                value, policy = self.evaluator.evaluate(search_board)
                node.expand(policy)
            else:
                value = self._get_game_outcome(search_board)

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
    Supports high-performance shared memory communication with EvaluationManager.
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
        shared_memory_config: Optional["SharedMemoryConfig"] = None,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.task_q = task_q
        self.result_q = result_q
        self.request_q = request_q
        self.response_q = response_q
        self.c_puct = c_puct
        self.n_scl = n_scl
        self.shared_memory_config = shared_memory_config

    def run(self):
        """Main loop: get task, run MCTS, put result."""
        evaluator = RemoteEvaluator(
            self.worker_id, self.request_q, self.response_q, self.shared_memory_config
        )
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
