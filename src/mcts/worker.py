from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Protocol, TYPE_CHECKING, List
import chess
import numpy as np
import logging
from dataclasses import dataclass
import threading
import time
import cProfile
import pstats
import queue

from node import MCTSNode
from src.encoder import Encoder
from src.move_mapping import index_to_move, ACTION_SPACE_SIZE, move_to_index
from src.utils import unmirror_policy

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
    q_value: Optional[float] = None
    error: Optional[str] = None
    worker_id: Optional[int] = None


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
        total_workers: int = 1,
    ):
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self.total_workers = total_workers
        logger.info(f"WORKER {worker_id} RemoteEvaluator received response queue {id(response_q)}")
        self.encoder = Encoder()
        self.shared_memory_config = shared_memory_config

        self.shared_memory_blocks: Dict[str, SharedMemory] = {}
        self.input_arrays: Dict[str, np.ndarray] = {}
        self.output_arrays: Dict[str, np.ndarray] = {}
        self.buffer_lock = threading.Lock()
        self.next_buffer_index = 0
        self.worker_buffer_names: List[str] = []
        self.worker_buffer_indices: List[int] = []

        if self.shared_memory_config:
            self._setup_shared_memory()

    def _setup_shared_memory(self):
        """Attach to existing shared memory blocks and partition them per worker."""
        logger.info(
            f"Worker {self.worker_id}: Attaching to {len(self.shared_memory_config.buffer_names)} total shared memory blocks..."
        )

        all_buffers = self.shared_memory_config.buffer_names
        base_per_worker = len(all_buffers) // self.total_workers
        remainder = len(all_buffers) % self.total_workers

        start_index = self.worker_id * base_per_worker + min(self.worker_id, remainder)
        num_buffers_for_worker = base_per_worker + (1 if self.worker_id < remainder else 0)
        end_index = start_index + num_buffers_for_worker

        if num_buffers_for_worker == 0:
            raise RuntimeError(f"Worker {self.worker_id} assigned 0 buffers. Increase `buffer_count` or decrease `num_workers`.")

        self.worker_buffer_names = all_buffers[start_index:end_index]
        self.worker_buffer_indices = list(range(start_index, end_index))

        logger.info(f"Worker {self.worker_id} assigned {len(self.worker_buffer_names)} private buffers (global indices {start_index}-{end_index-1}).")

        for buffer_name in self.worker_buffer_names:
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
        Allocate a buffer from this worker's private pool.
        Uses a simple round-robin allocation strategy within its own slice.
        """
        with self.buffer_lock:
            local_buffer_idx = self.next_buffer_index % len(self.worker_buffer_names)
            self.next_buffer_index += 1
            global_buffer_idx = self.worker_buffer_indices[local_buffer_idx]
            return global_buffer_idx

    def _write_input_to_buffer(self, buffer_index: int, encoded_state: np.ndarray):
        """Write input data to a shared memory buffer."""
        buffer_name = self.worker_buffer_names[buffer_index % len(self.worker_buffer_names)]
        input_array = self.input_arrays[buffer_name]
        input_array[:] = encoded_state

    def _read_output_from_buffer(self, buffer_index: int) -> tuple[float, np.ndarray]:
        """Read output data from a shared memory buffer."""
        buffer_name = self.worker_buffer_names[buffer_index % len(self.worker_buffer_names)]
        output_array = self.output_arrays[buffer_name]

        # The model's value head outputs a 3-element vector.
        value_probs = output_array[0:3]

        # Calculate the expected value: (win * 1) + (draw * 0) + (loss * -1)
        # Assuming the model's output softmax is in the order [loss, draw, win]
        value = float(value_probs[2] - value_probs[0])

        # The policy logits start AFTER the 3 value probabilities.
        policy_logits = output_array[3:].copy()

        return value, policy_logits

    def queue_evaluation(self, board: chess.Board) -> Dict:
        """
        Queues an evaluation request and returns a handle without blocking.
        The handle contains the information needed to retrieve the result later.
        """
        if not self.shared_memory_config:
            raise RuntimeError("RemoteEvaluator requires shared memory.")

        # Mirror board for Black so the model always sees White's perspective
        is_black = board.turn == chess.BLACK
        eval_board = board.mirror() if is_black else board

        encoded_state = self.encoder.encode(eval_board)
        buffer_index = self._allocate_buffer()
        self._write_input_to_buffer(buffer_index, encoded_state)

        request = {"worker_id": self.worker_id, "buffer_index": buffer_index}
        self.request_q.put(request)

        return {
            "buffer_index": buffer_index,
            "fen": board.fen(),
            "is_black": is_black
        }

    def collect_evaluation_batch(self, handles: List[Dict]) -> List[tuple]:
        """
        Collects a batch of evaluation results corresponding to the provided handles.
        """
        results_map = {}
        for _ in range(len(handles)):
            try:
                response = self.response_q.get(timeout=300.0)
                if "error" in response:
                    raise RuntimeError(f"Evaluation failed in manager: {response['error']}")

                buffer_idx = response["buffer_index"]
                value, policy_logits = self._read_output_from_buffer(buffer_idx)
                results_map[buffer_idx] = (value, policy_logits)
            except queue.Empty:
                raise RuntimeError(f"Worker {self.worker_id} timed out waiting for evaluation response.")

        ordered_results = []
        for handle in handles:
            buffer_idx = handle["buffer_index"]
            board = chess.Board(handle["fen"])
            value, policy_logits = results_map[buffer_idx]
            
            # If we mirrored the board input (Black), we must unmirror the policy output
            if handle.get("is_black", False):
                policy_logits = unmirror_policy(policy_logits)
            
            policy = self._decode_policy(policy_logits, board)
            ordered_results.append((value, policy))

        return ordered_results

    def evaluate(self, board: chess.Board) -> tuple[float, Dict[chess.Move, float]]:
        """
        Legacy synchronous evaluation. For compatibility if needed, but the new
        pattern is queue_evaluation followed by collect_evaluation_batch.
        """
        handle = self.queue_evaluation(board)
        return self.collect_evaluation_batch([handle])[0]

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

        mask = np.ones(logits.shape, dtype=bool)
        mask[legal_move_indices] = False
        logits[mask] = -np.inf  # Apply mask for illegal moves

        # Numerically stable softmax using NumPy
        # Subtract the max for stability before exponentiating
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        policy_dict = {
            move: float(probabilities[move_to_index(move, board)])
            for move in legal_moves
            if move_to_index(move, board) is not None
        }

        # Renormalize just in case of floating point inaccuracies
        total_prob = sum(policy_dict.values())
        if total_prob > 0:
            return {move: prob / total_prob for move, prob in policy_dict.items()}
        return {}

    def cleanup(self):
        """Clean up shared memory handles before worker exit."""
        logger.debug(f"Worker {self.worker_id} cleaning up its shared memory handles...")
        # Clear numpy array views first to remove references
        self.input_arrays.clear()
        self.output_arrays.clear()
        # Then close (but don't unlink) the shared memory blocks
        for shm in self.shared_memory_blocks.values():
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"Error closing shared memory in worker {self.worker_id}: {e}")
        self.shared_memory_blocks.clear()


class MCTS:
    """
    Encapsulates the core Monte Carlo Tree Search algorithm.
    """

    def __init__(self, evaluator: Evaluator, c_puct: float, n_scl: int):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_scl = n_scl

        num_private_buffers = 0
        if hasattr(self.evaluator, 'worker_buffer_names'):
             num_private_buffers = len(self.evaluator.worker_buffer_names)

        self.local_batch_size = min(8, num_private_buffers)
        if self.local_batch_size == 0:
            logger.warning(f"MCTS worker has {num_private_buffers} private buffers, pipelining will be disabled.")

    def run_search(
        self, board: chess.Board, num_simulations: int
    ) -> tuple[Dict[chess.Move, float], float]:
        """
        Performs MCTS search using worker-side batching to pipeline evaluations.
        Returns the final policy and the root node's Q-value.
        """
        root = MCTSNode(depth=0)

        if not hasattr(self.evaluator, 'queue_evaluation') or not hasattr(self.evaluator, 'collect_evaluation_batch') or self.local_batch_size == 0:
            logger.warning("Falling back to synchronous (non-pipelined) search.")
            for _ in range(num_simulations):
                node = root
                search_board = board.copy()
                while not node.is_leaf():
                    move, node = node.select_child(self.c_puct, self.n_scl)
                    search_board.push(move)
                value, policy = self.evaluator.evaluate(search_board)
                node.expand(policy)
                node.update(value)
            policy = self._calculate_final_policy(root)
            q_value = root.q_value()
            return policy, q_value

        sims_processed = 0
        while sims_processed < num_simulations:
            pending_evaluations = []

            for _ in range(min(self.local_batch_size, num_simulations - sims_processed)):
                node = root
                search_board = board.copy()

                while not node.is_leaf():
                    child_data = node.select_child(self.c_puct, self.n_scl)
                    if child_data is None:
                        break
                    move, node = child_data
                    search_board.push(move)

                if search_board.is_game_over(claim_draw=True):
                    value = self._get_game_outcome(search_board)
                    node.update(value)
                    sims_processed += 1
                else:
                    handle = self.evaluator.queue_evaluation(search_board)
                    pending_evaluations.append({'handle': handle, 'node': node})

            if not pending_evaluations:
                if sims_processed >= num_simulations:
                    break
                else:
                    continue

            try:
                results = self.evaluator.collect_evaluation_batch(
                    [p['handle'] for p in pending_evaluations]
                )

                for pending_item, result_data in zip(pending_evaluations, results):
                    node = pending_item['node']
                    value, policy = result_data

                    if node.visit_count == 0:
                        node.expand(policy)

                    node.update(value)
                    sims_processed += 1
            except RuntimeError as e:
                logger.error(f"Worker {self.evaluator.worker_id} failed during batch collection: {e}")
                sims_processed += len(pending_evaluations)

        policy = self._calculate_final_policy(root)
        q_value = root.q_value()
        return policy, q_value

    def _get_game_outcome(self, board: chess.Board) -> float:
        """Determines the game outcome from the perspective of the current player."""
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _calculate_final_policy(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """
        Calculates the final policy from the raw visit counts of the root's children.
        Instead of probabilities, this returns the integer visit counts to preserve
        full resolution for aggregation in the controller.
        """
        if root.is_leaf() or root.visit_count == 0:
            return {}

        return {
            move: child.visit_count
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
        total_workers: int,
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
        self.total_workers = total_workers
        self.c_puct = c_puct
        self.n_scl = n_scl
        self.shared_memory_config = shared_memory_config

    def run(self):
        """The main loop for the worker process."""
        logger.info(f"WORKER {self.worker_id}: Starting main loop.")
        profiler = None
        if self.worker_id == 0:
            print("INFO: Profiler enabled for worker 0.")
            profiler = cProfile.Profile()
            profiler.enable()

        evaluator = RemoteEvaluator(
            self.worker_id, self.request_q, self.response_q, self.shared_memory_config,
            total_workers=self.total_workers
        )
        mcts_instance = MCTS(evaluator, self.c_puct, self.n_scl)

        while True:
            task: Optional[SearchTask] = self.task_q.get()
            if task is None:
                logger.info(f"Worker {self.worker_id} shutting down.")
                break

            try:
                board = chess.Board(task.fen)
                policy, q_value = mcts_instance.run_search(board, task.num_simulations)
                result = SearchResult(fen=task.fen, policy=policy, q_value=q_value, worker_id=self.worker_id)
                self.result_q.put(result)
                logger.info(f"WORKER {self.worker_id}: Finished search for FEN {task.fen} and sent result.")
            except Exception as e:
                logger.error(
                    f"Error in worker {self.worker_id} for FEN {task.fen}: {e}",
                    exc_info=True,
                )
                result = SearchResult(fen=task.fen, policy={}, error=str(e), worker_id=self.worker_id)
                self.result_q.put(result)

        if profiler:
            profiler.disable()
            stats_file = "worker_0_profile.prof"
            profiler.dump_stats(stats_file)
            print(f"INFO: Worker 0 profiling data saved to '{stats_file}'")

        evaluator.cleanup()
        logger.info(f"Worker {self.worker_id} shutting down cleanly.")
