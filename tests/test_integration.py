import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from multiprocessing import Queue, Process, set_start_method
import time
import chess
import numpy as np
import tensorflow as tf

from src.mcts.worker import SearchWorker, SearchTask, SearchResult
from src.mcts.manager import EvaluationManager


class TestMCTSIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set the start method to 'spawn' and force CPU usage for consistency.
        This is crucial for compatibility with TensorFlow/CUDA in multiprocessing
        and ensures tests run reliably on different hardware.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            set_start_method("spawn")
        except RuntimeError:
            # The start method can only be set once. If it's already been set,
            # we can ignore the error.
            pass

    def setUp(self):
        """Set up queues and processes before each test."""
        self.task_q = Queue()
        self.result_q = Queue()
        self.request_q = Queue()
        # Create a list of response queues for the new API
        self.response_qs: list[Queue] = []
        self.processes: list[Process] = []

    def tearDown(self):
        """Clean up queues and processes after each test."""
        # Terminate all running processes
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # Close queues
        all_queues = [self.task_q, self.result_q, self.request_q] + self.response_qs
        for q in all_queues:
            q.close()
            q.join_thread()

    def _start_manager(
        self, num_workers=1, batch_size=4, wait_ms=10, use_error_model=False
    ):
        """Starts a manager with a mock model for predictable testing."""
        # The manager now needs a list of response queues
        self.response_qs = [Queue() for _ in range(num_workers)]
        manager = EvaluationManager(
            request_q=self.request_q,
            response_qs=self.response_qs,
            weights_path="",  # Not used with mock models
            batch_size=batch_size,
            max_wait_time_ms=wait_ms,
            use_error_model=use_error_model,
            use_mock_model=not use_error_model,  # Use mock unless error model is specified
        )
        manager.start()
        self.processes.append(manager)
        return manager

    def _start_workers(self, num_workers=1):
        # Ensure there are enough response queues for the workers
        if not self.response_qs or len(self.response_qs) != num_workers:
            self.response_qs = [Queue() for _ in range(num_workers)]

        for i in range(num_workers):
            worker = SearchWorker(
                worker_id=i,
                task_q=self.task_q,
                result_q=self.result_q,
                request_q=self.request_q,
                response_q=self.response_qs[i],  # Each worker gets its own queue
            )
            worker.start()
            self.processes.append(worker)

    def test_single_worker_and_manager(self):
        """
        Happy Path: Test that a single worker and manager can complete a search task.
        """
        self._start_manager(num_workers=1)
        self._start_workers(1)

        start_fen = chess.STARTING_FEN
        self.task_q.put(SearchTask(fen=start_fen, num_simulations=5))

        # Wait for the result
        result: SearchResult = self.result_q.get(timeout=10)

        self.assertIsNone(result.error)
        self.assertEqual(result.fen, start_fen)
        self.assertIsInstance(result.policy, dict)
        self.assertGreater(len(result.policy), 0)

    def test_multiple_workers(self):
        """
        Concurrency Test: Verify the system works with multiple workers.
        """
        num_workers = 4
        self._start_manager(num_workers=num_workers, batch_size=num_workers)
        self._start_workers(num_workers)

        start_fen = chess.STARTING_FEN
        for _ in range(num_workers):
            self.task_q.put(SearchTask(fen=start_fen, num_simulations=5))

        results = []
        for _ in range(num_workers):
            result: SearchResult = self.result_q.get(timeout=30)
            self.assertIsNone(result.error)
            results.append(result)

        self.assertEqual(len(results), num_workers)

    def test_graceful_shutdown(self):
        """
        Lifecycle Test: Ensure workers and manager shut down cleanly.
        """
        manager = self._start_manager(num_workers=2)
        self._start_workers(2)

        # Send sentinel value to terminate workers
        self.task_q.put(None)
        self.task_q.put(None)

        # Wait for workers to finish
        worker_processes = [p for p in self.processes if isinstance(p, SearchWorker)]
        for p in worker_processes:
            p.join(timeout=5)
            self.assertFalse(p.is_alive())

        # Now that workers are done, the manager can be shut down.
        # It's better to terminate it explicitly for test robustness.
        manager.terminate()
        manager.join(timeout=5)
        self.assertFalse(manager.is_alive())

    def test_error_propagation_from_model(self):
        """
        Edge Case: Test that an error during inference is propagated back to the worker.
        """
        # Start a manager specifically configured to use an error-raising model
        self._start_manager(num_workers=1, use_error_model=True)
        self._start_workers(1)

        self.task_q.put(SearchTask(fen=chess.STARTING_FEN, num_simulations=5))

        result: SearchResult = self.result_q.get(timeout=10)

        self.assertIsNotNone(result.error)
        self.assertIn("Test Inference Error", result.error)
        self.assertEqual(result.policy, {})

    def test_mcts_controller_migration_example(self):
        """
        Example test showing migration from manual process management to MCTSController.
        This demonstrates the simplified API.
        """
        from src.mcts.controller import MCTSController

        # New simplified approach using MCTSController
        with MCTSController(
            num_workers=2,
            use_mock_model=True,
            buffer_count=16,
        ) as controller:
            result = controller.run_search(chess.STARTING_FEN, num_simulations=10)

            self.assertIsNone(result.error)
            self.assertEqual(result.fen, chess.STARTING_FEN)
            self.assertIsInstance(result.policy, dict)
            self.assertGreater(len(result.policy), 0)


if __name__ == "__main__":
    unittest.main()
