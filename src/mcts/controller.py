"""
High-performance MCTS Controller with shared memory IPC.

This module implements the controller for managing a multi-process MCTS engine
that uses shared memory for high-throughput data transfer between SearchWorkers
and the EvaluationManager, while using lightweight queues for coordination.
"""

import logging
import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
from dataclasses import dataclass, field
import queue
import threading

from src.mcts.manager import EvaluationManager
from src.mcts.worker import SearchWorker, SearchTask, SearchResult

logger = logging.getLogger(__name__)

# --- MODIFICATION: Create a new result class for the final aggregated data ---
@dataclass
class MCTSResult:
    """Represents the final, aggregated result of an MCTS search."""
    policy: Dict[Any, float]
    q_value: float
    error: Optional[str] = None

# Configuration constants
DEFAULT_BUFFER_COUNT = 256
INPUT_BUFFER_SIZE = 8 * 8 * 34 * 4  # (8,8,34) float32 = 8704 bytes
OUTPUT_BUFFER_SIZE = (
    3 + 4672
) * 4  # value (3) + policy_logits (4672) float32 = 18700 bytes
TOTAL_BUFFER_SIZE = INPUT_BUFFER_SIZE + OUTPUT_BUFFER_SIZE  # 27404 bytes per buffer


@dataclass
class SharedMemoryConfig:
    """Configuration for shared memory buffers."""

    buffer_names: List[str] = field(default_factory=list)
    buffer_count: int = DEFAULT_BUFFER_COUNT
    input_shape: Tuple[int, ...] = (8, 8, 34)
    policy_size: int = 4672
    input_dtype: np.dtype = np.float32
    output_dtype: np.dtype = np.float32

    def get_input_size(self) -> int:
        """Get the size in bytes for input data."""
        return int(np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize)

    def get_output_size(self) -> int:
        """Get the size in bytes for output data (value + policy)."""
        return int((3 + self.policy_size) * np.dtype(self.output_dtype).itemsize)

    def get_total_size(self) -> int:
        """Get the total size in bytes for one buffer."""
        return self.get_input_size() + self.get_output_size()


class MCTSController:
    """
    Central controller for high-performance MCTS with shared memory IPC.

    This controller manages the entire lifecycle of a multiprocess MCTS search:
    - Creates and manages shared memory pools for high-throughput data transfer
    - Coordinates SearchWorker processes and EvaluationManager process
    - Provides a clean interface for running searches

    The architecture uses a hybrid IPC approach:
    - Heavy data (NumPy arrays) transferred via shared memory (zero-copy)
    - Lightweight coordination via multiprocessing.Queue
    """

    def __init__(
        self,
        num_workers: int = 4,
        model_weights_path: str = None,
        batch_size: int = 32,
        max_wait_time_ms: float = 10.0,
        buffer_count: int = DEFAULT_BUFFER_COUNT,
        use_mock_model: bool = False,
        use_error_model: bool = False,
    ):
        """
        Initialize the MCTS Controller.

        Args:
            num_workers: Number of SearchWorker processes to spawn
            model_weights_path: Path to neural network model weights
            batch_size: Batch size for neural network inference
            max_wait_time_ms: Max wait time for batching requests
            buffer_count: Number of shared memory buffers to create
            use_mock_model: Use mock model for testing
            use_error_model: Use error-raising model for testing
        """
        self.num_workers = num_workers
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.buffer_count = buffer_count
        self.use_mock_model = use_mock_model
        self.use_error_model = use_error_model

        # IPC components
        self.task_q: Optional[Queue] = None
        self.result_q: Optional[Queue] = None
        self.request_q: Optional[Queue] = None
        self.manager_response_q: Optional[Queue] = None
        self.response_qs: List[Queue] = []

        # Process management
        self.workers: List[SearchWorker] = []
        self.evaluation_manager: Optional[EvaluationManager] = None

        # Shared memory management
        self.shared_memory_blocks: List[SharedMemory] = []
        self.shared_memory_config: Optional[SharedMemoryConfig] = None

        self._is_started = False

    def _setup_ipc(self) -> None:
        """
        Set up Inter-Process Communication infrastructure.

        Creates:
        - Control plane: Lightweight queues for coordination
        - Data plane: Shared memory pool for high-throughput data transfer
        """
        logger.info("Setting up IPC infrastructure...")

        # Create control queues
        self.task_q = Queue()
        self.result_q = Queue()
        self.request_q = Queue()
        self.manager_response_q = Queue()
        self.response_qs = [Queue() for _ in range(self.num_workers)]
        logger.info(f"Created {len(self.response_qs)} dedicated response queues.")

        # Create shared memory pool
        self._create_shared_memory_pool()

        logger.info(
            f"IPC setup complete: {len(self.shared_memory_blocks)} shared memory buffers created"
        )

    def _create_shared_memory_pool(self) -> None:
        """
        Create a pool of shared memory blocks for data transfer.

        Each buffer contains space for:
        - Input: encoded board state (8,8,34) float32
        - Output: value (3 floats) + policy logits (4672 floats)
        """
        self.shared_memory_config = SharedMemoryConfig(buffer_count=self.buffer_count)
        buffer_size = self.shared_memory_config.get_total_size()

        logger.info(
            f"Creating {self.buffer_count} shared memory buffers of {buffer_size} bytes each"
        )

        for i in range(self.buffer_count):
            try:
                # Create shared memory block
                shm = SharedMemory(create=True, size=buffer_size)
                self.shared_memory_blocks.append(shm)
                self.shared_memory_config.buffer_names.append(shm.name)

                # Initialize buffer to zeros
                buffer = np.frombuffer(shm.buf, dtype=np.uint8)
                buffer.fill(0)

            except Exception as e:
                logger.error(f"Failed to create shared memory buffer {i}: {e}")
                self._cleanup_shared_memory()
                raise RuntimeError(f"Shared memory setup failed: {e}")

        logger.info(
            f"Successfully created {len(self.shared_memory_blocks)} shared memory buffers"
        )

    def start(self) -> None:
        """
        Start the MCTS engine by launching all worker processes.

        This method:
        1. Sets up IPC infrastructure
        2. Starts the EvaluationManager process
        3. Starts all SearchWorker processes
        """
        if self._is_started:
            logger.warning("MCTS Controller is already started")
            return

        try:
            # Setup IPC
            self._setup_ipc()

            # Start the response router thread
            self.router_thread = threading.Thread(
                target=self._response_router_thread, daemon=True
            )
            self.router_thread.start()

            # Start EvaluationManager
            logger.info("Starting EvaluationManager process...")
            self.evaluation_manager = EvaluationManager(
                request_q=self.request_q,
                response_q=self.manager_response_q,
                weights_path=self.model_weights_path,
                batch_size=self.batch_size,
                max_wait_time_ms=self.max_wait_time_ms,
                use_mock_model=self.use_mock_model,
                use_error_model=self.use_error_model,
                shared_memory_config=self.shared_memory_config,
            )
            self.evaluation_manager.start()

            # Start SearchWorker processes
            logger.info(f"Starting {self.num_workers} SearchWorker processes...")
            for worker_id in range(self.num_workers):
                response_q = self.response_qs[worker_id]
                logger.info(f"Assigning response queue {id(response_q)} to Worker {worker_id}")
                worker = SearchWorker(
                    worker_id=worker_id,
                    task_q=self.task_q,
                    result_q=self.result_q,
                    request_q=self.request_q,
                    response_q=response_q,
                    total_workers=self.num_workers,
                    c_puct=4.0,
                    n_scl=100_000,
                    shared_memory_config=self.shared_memory_config,
                )
                worker.start()
                self.workers.append(worker)

            self._is_started = True
            logger.info(
                f"MCTS Controller started successfully with {self.num_workers} workers"
            )

        except Exception as e:
            logger.error(f"Failed to start MCTS Controller: {e}")
            self.shutdown()
            raise

    def run_search(self, fen: str, num_simulations: int) -> MCTSResult:
        """
        Run an MCTS search by distributing simulations across all workers.

        This implements a parallel search where each worker runs a fraction of
        the total simulations on an independent tree. The results (visit counts and q_values)
        are then aggregated to form the final policy and value estimate.

        Args:
            fen: Chess position in FEN notation
            num_simulations: Total number of MCTS simulations to run across all workers.

        Returns:
            A single MCTSResult containing the aggregated policy and Q-value.
        """
        if not self._is_started:
            raise RuntimeError("MCTS Controller must be started before running searches")

        if self.num_workers == 0:
            logger.warning("run_search called with zero workers.")
            return MCTSResult(policy={}, q_value=0.0, error="No workers available.")

        # --- Distribute work evenly among all workers ---
        sims_per_worker = max(1, num_simulations // self.num_workers)
        actual_total_sims = sims_per_worker * self.num_workers

        logger.info(
            f"Dispatching {self.num_workers} tasks, {sims_per_worker} simulations each (total: {actual_total_sims})."
        )

        for _ in range(self.num_workers):
            task = SearchTask(fen=fen, num_simulations=sims_per_worker)
            self.task_q.put(task)

        # --- Collect and aggregate results from all workers ---
        aggregated_policy: Dict[Any, float] = {}
        aggregated_q_value: float = 0.0
        num_valid_results: int = 0
        errors: List[str] = []

        for i in range(self.num_workers):
            try:
                # Use a generous timeout, as this worker might be waiting for others.
                result: SearchResult = self.result_q.get(timeout=600.0)

                if result.error:
                    errors.append(f"Worker {result.worker_id} Error: {result.error}")
                    continue

                # --- MODIFICATION: Aggregate raw visit counts and q_values ---
                num_valid_results += 1
                if result.q_value is not None:
                    aggregated_q_value += result.q_value

                for move, visit_count in result.policy.items():
                    aggregated_policy[move] = aggregated_policy.get(move, 0.0) + visit_count

            except queue.Empty:
                error_msg = f"Timeout: Only received {i}/{self.num_workers} results."
                logger.error(error_msg)
                errors.append(error_msg)
                break  # Stop waiting if one worker fails to report back
            except Exception as e:
                errors.append(f"An unexpected error occurred while collecting results: {e}")

        # --- Finalize the aggregated policy and value ---
        if num_valid_results == 0:
            final_error = " | ".join(errors) if errors else "Policy and value aggregation failed: no results."
            return MCTSResult(policy={}, q_value=0.0, error=final_error)

        final_q_value = aggregated_q_value / num_valid_results if num_valid_results > 0 else 0.0

        total_visits = sum(aggregated_policy.values())
        if total_visits > 0:
            final_policy = {move: visits / total_visits for move, visits in aggregated_policy.items()}
        else:
            final_policy = {}

        final_error = " | ".join(errors) if errors else None
        return MCTSResult(policy=final_policy, q_value=final_q_value, error=final_error)

    def shutdown(self) -> None:
        """
        Cleanly shut down all processes and shared memory.
        """
        if not self._is_started:
            return

        logger.info("Shutting down MCTS Controller...")

        # Terminate SearchWorker processes
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
        self.workers.clear()

        # Signal EvaluationManager to shut down
        if self.request_q:
            try:
                self.request_q.put(None)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to manager: {e}")

        # Signal the Response Router thread to shut down
        if self.manager_response_q:
            try:
                self.manager_response_q.put(None)
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to router: {e}")

        # Join EvaluationManager
        if self.evaluation_manager and self.evaluation_manager.is_alive():
            logger.debug("Waiting for EvaluationManager to terminate...")
            self.evaluation_manager.join(timeout=5.0)

        # Join the router thread
        if hasattr(self, 'router_thread') and self.router_thread.is_alive():
            logger.debug("Waiting for Response Router thread to terminate...")
            self.router_thread.join(timeout=2.0)

        # Cleanup IPC resources
        self._cleanup_shared_memory()

        if self.task_q: self.task_q.close()
        if self.result_q: self.result_q.close()
        if self.request_q: self.request_q.close()
        if self.manager_response_q: self.manager_response_q.close()
        for q in self.response_qs:
            q.close()

        self._is_started = False
        logger.info("MCTS Controller shut down successfully.")

    def _cleanup_shared_memory(self) -> None:
        """
        Close and unlink all shared memory blocks.
        """
        logger.info(
            f"Cleaning up {len(self.shared_memory_blocks)} shared memory blocks..."
        )

        for shm in self.shared_memory_blocks:
            try:
                shm.close()
                shm.unlink()  # Free up the memory
            except FileNotFoundError:
                pass  # It might have been unlinked already
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory: {e}")
        self.shared_memory_blocks.clear()
        if self.shared_memory_config:
            self.shared_memory_config.buffer_names.clear()

    def _response_router_thread(self):
        """
        Continuously pulls results from the manager's central response queue
        and routes them to the correct worker's dedicated response queue.
        This is a daemon thread that runs in the background.
        """
        logger.info("[Router] Response router thread started.")
        while True:
            try:
                # Block and wait for a response from the manager
                response = self.manager_response_q.get()

                # Sentinel value to terminate the thread
                if response is None:
                    logger.info("[Router] Shutdown signal received. Exiting.")
                    break

                # Get the destination worker's ID
                worker_id = response.get("worker_id")
                if worker_id is not None and 0 <= worker_id < self.num_workers:
                    # Put the response onto that specific worker's queue
                    self.response_qs[worker_id].put(response)
                else:
                    logger.warning(f"[Router] Received response with invalid worker_id: {response}")

            except Exception as e:
                logger.error(f"[Router] Error in response router thread: {e}", exc_info=True)
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.shutdown()
