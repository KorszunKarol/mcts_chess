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

from src.mcts.manager import EvaluationManager
from src.mcts.worker import SearchWorker, SearchTask, SearchResult

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BUFFER_COUNT = 256
INPUT_BUFFER_SIZE = 8 * 8 * 34 * 4  # (8,8,34) float32 = 8704 bytes
OUTPUT_BUFFER_SIZE = (
    1 + 4672
) * 4  # value (1) + policy_logits (4672) float32 = 18692 bytes
TOTAL_BUFFER_SIZE = INPUT_BUFFER_SIZE + OUTPUT_BUFFER_SIZE  # 27396 bytes per buffer


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
        return int((1 + self.policy_size) * np.dtype(self.output_dtype).itemsize)

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
        self.response_qs = [Queue() for _ in range(self.num_workers)]

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
        - Output: value (1 float) + policy logits (4672 floats)
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

            # Start EvaluationManager
            logger.info("Starting EvaluationManager process...")
            self.evaluation_manager = EvaluationManager(
                request_q=self.request_q,
                response_qs=self.response_qs,
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
                worker = SearchWorker(
                    worker_id=worker_id,
                    task_q=self.task_q,
                    result_q=self.result_q,
                    request_q=self.request_q,
                    response_q=self.response_qs[worker_id],
                    c_puct=1.0,
                    n_scl=1000,
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

    def run_search(self, fen: str, num_simulations: int) -> SearchResult:
        """
        Run an MCTS search for the given position.

        Args:
            fen: Chess position in FEN notation
            num_simulations: Number of MCTS simulations to run

        Returns:
            SearchResult containing the policy and any errors
        """
        if not self._is_started:
            raise RuntimeError(
                "MCTS Controller must be started before running searches"
            )

        # Create and submit search task
        task = SearchTask(fen=fen, num_simulations=num_simulations)
        logger.debug(
            f"Submitting search task: {fen} with {num_simulations} simulations"
        )

        self.task_q.put(task)

        # Wait for result
        try:
            result = self.result_q.get(timeout=60.0)  # 60 second timeout
            logger.debug(f"Received search result for {fen}")
            return result
        except Exception as e:
            logger.error(f"Failed to get search result: {e}")
            return SearchResult(fen=fen, policy={}, error=str(e))

    def shutdown(self) -> None:
        """
        Gracefully shutdown the MCTS engine.

        This method:
        1. Terminates all worker processes
        2. Terminates the EvaluationManager process
        3. Cleans up shared memory blocks
        """
        if not self._is_started:
            logger.info("MCTS Controller is not started, nothing to shutdown")
            return

        logger.info("Shutting down MCTS Controller...")

        # Terminate worker processes by sending sentinel value
        if self.task_q:
            for _ in self.workers:
                try:
                    self.task_q.put(None)
                except Exception as e:
                    logger.warning(f"Could not send sentinel to task queue: {e}")

        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    logger.warning(
                        f"Worker {worker.pid} did not terminate gracefully, forcing termination."
                    )
                    worker.terminate()

        # Signal EvaluationManager to shut down
        if self.request_q:
            try:
                # Add a sentinel value to unblock the manager's queue
                self.request_q.put(None)
            except Exception as e:
                logger.error(f"Error putting sentinel on request_q: {e}")

        # Wait for EvaluationManager to terminate
        if self.evaluation_manager and self.evaluation_manager.is_alive():
            logger.debug("Waiting for EvaluationManager to terminate...")
            self.evaluation_manager.join(timeout=5)
            if self.evaluation_manager.is_alive():
                logger.warning("EvaluationManager did not terminate, forcing.")
                self.evaluation_manager.terminate()

        # Clean up shared memory
        self._cleanup_shared_memory()

        self._is_started = False
        logger.info("MCTS Controller shutdown complete")

    def _cleanup_shared_memory(self) -> None:
        """Clean up all shared memory blocks."""
        logger.info(
            f"Cleaning up {len(self.shared_memory_blocks)} shared memory blocks..."
        )

        for shm in self.shared_memory_blocks:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup shared memory {shm.name}: {e}")

        self.shared_memory_blocks.clear()
        if self.shared_memory_config:
            self.shared_memory_config.buffer_names.clear()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.shutdown()
