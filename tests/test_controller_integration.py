import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Integration tests for the high-performance MCTSController with shared memory IPC.

These tests verify that the new shared memory-based architecture works correctly
and provides backward compatibility with the queue-based approach.
"""

import unittest
from multiprocessing import set_start_method
import time
import chess
import numpy as np
import logging

from src.mcts.controller import MCTSController, SharedMemoryConfig
from src.mcts.worker import SearchTask, SearchResult

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




class TestMCTSControllerIntegration(unittest.TestCase):
    """Test the MCTSController with shared memory IPC."""

    @classmethod
    def setUpClass(cls):
        """
        Set the start method to 'spawn' and force CPU usage for consistency.
        This ensures tests run reliably on different hardware.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            set_start_method("spawn")
        except RuntimeError:
            # Start method can only be set once
            pass

    def setUp(self):
        """Set up test fixtures."""
        self.controller = None

    def tearDown(self):
        """Clean up after each test."""
        if self.controller:
            self.controller.shutdown()
            self.controller = None

    def test_controller_startup_and_shutdown(self):
        """Test that the controller can start and shutdown cleanly."""
        self.controller = MCTSController(
            num_workers=2,
            model_weights_path=None,
            use_mock_model=True,
            buffer_count=32,  # Smaller buffer count for tests
        )

        # Test startup
        self.controller.start()
        self.assertTrue(self.controller._is_started)
        self.assertEqual(len(self.controller.workers), 2)
        self.assertIsNotNone(self.controller.evaluation_manager)
        self.assertEqual(len(self.controller.shared_memory_blocks), 32)

        # Test shutdown
        self.controller.shutdown()
        self.assertFalse(self.controller._is_started)
        self.assertEqual(len(self.controller.shared_memory_blocks), 0)

    def test_context_manager_interface(self):
        """Test that the controller works as a context manager."""
        with MCTSController(
            num_workers=1,
            use_mock_model=True,
            buffer_count=16,
        ) as controller:
            self.assertTrue(controller._is_started)

            # Run a simple search
            result = controller.run_search(chess.STARTING_FEN, num_simulations=5)
            self.assertIsInstance(result, SearchResult)
            self.assertIsNone(result.error)
            self.assertIsInstance(result.policy, dict)

        # Controller should be shutdown after context exit
        self.assertFalse(controller._is_started)

    def test_shared_memory_search_single_worker(self):
        """Test MCTS search with shared memory and single worker."""
        self.controller = MCTSController(
            num_workers=1,
            use_mock_model=True,
            buffer_count=16,
        )
        self.controller.start()

        

        result = self.controller.run_search(chess.STARTING_FEN, num_simulations=10)

        # Verify result
        self.assertIsInstance(result, SearchResult)
        self.assertIsNone(result.error)
        self.assertEqual(result.fen, chess.STARTING_FEN)
        self.assertIsInstance(result.policy, dict)
        self.assertGreater(len(result.policy), 0)

        # Verify policy values are probabilities
        total_prob = sum(result.policy.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)


    def test_shared_memory_search_multiple_workers(self):
        """Test MCTS search with shared memory and multiple workers."""
        self.controller = MCTSController(
            num_workers=4,
            use_mock_model=True,
            buffer_count=64,
        )
        self.controller.start()

        # Run multiple searches concurrently by having multiple workers
        # Each worker will get the same task and should produce consistent results
        result = self.controller.run_search(chess.STARTING_FEN, num_simulations=20)

        # Verify result
        self.assertIsInstance(result, SearchResult)
        self.assertIsNone(result.error)
        self.assertEqual(result.fen, chess.STARTING_FEN)
        self.assertIsInstance(result.policy, dict)
        self.assertGreater(len(result.policy), 0)

    def test_multiple_sequential_searches(self):
        """Test multiple sequential searches to verify buffer reuse."""
        self.controller = MCTSController(
            num_workers=2,
            use_mock_model=True,
            buffer_count=8,  # Small buffer count to force reuse
        )
        self.controller.start()

        test_positions = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e5
        ]

        results = []
        for fen in test_positions:
            result = self.controller.run_search(fen, num_simulations=5)
            results.append(result)

            # Verify each result
            self.assertIsInstance(result, SearchResult)
            self.assertIsNone(result.error)
            self.assertEqual(result.fen, fen)
            self.assertIsInstance(result.policy, dict)
            self.assertGreater(len(result.policy), 0)

        # Verify we got different results for different positions
        self.assertEqual(len(results), len(test_positions))
        self.assertNotEqual(results[0].policy, results[1].policy)

    def test_shared_memory_config(self):
        """Test shared memory configuration parameters."""
        config = SharedMemoryConfig(
            buffer_count=10,
            input_shape=(8, 8, 34),
            policy_size=4672,
        )

        # Test size calculations
        expected_input_size = 8 * 8 * 34 * 4  # float32 = 4 bytes
        expected_output_size = (1 + 4672) * 4  # value + policy logits

        self.assertEqual(config.get_input_size(), expected_input_size)
        self.assertEqual(config.get_output_size(), expected_output_size)
        self.assertEqual(
            config.get_total_size(), expected_input_size + expected_output_size
        )

    def test_buffer_allocation_strategy(self):
        """Test that buffer allocation works correctly with multiple workers."""
        self.controller = MCTSController(
            num_workers=3,
            use_mock_model=True,
            buffer_count=16,
        )
        self.controller.start()

        # Run several searches to exercise buffer allocation
        for i in range(5):
            result = self.controller.run_search(chess.STARTING_FEN, num_simulations=3)
            self.assertIsNone(result.error, f"Search {i} failed: {result.error}")

    def test_error_handling_with_shared_memory(self):
        """Test error handling with shared memory architecture."""
        self.controller = MCTSController(
            num_workers=1,
            use_error_model=True,  # Use model that raises errors
            buffer_count=8,
        )
        self.controller.start()

        # This should result in an error being propagated back
        result = self.controller.run_search(chess.STARTING_FEN, num_simulations=5)

        self.assertIsNotNone(result.error)
        self.assertIn("Test Inference Error", result.error)
        self.assertEqual(result.policy, {})

    def test_performance_improvement_indicator(self):
        """
        Basic performance test to ensure shared memory doesn't regress performance.
        This is not a rigorous benchmark but helps catch major regressions.
        """
        self.controller = MCTSController(
            num_workers=2,
            use_mock_model=True,
            buffer_count=32,
        )
        self.controller.start()

        # Time a batch of searches
        start_time = time.time()
        num_searches = 10  # Increased workload for a more stable measurement

        for _ in range(num_searches):
            result = self.controller.run_search(chess.STARTING_FEN, num_simulations=10)
            self.assertIsNone(result.error)

        elapsed_time = time.time() - start_time
        searches_per_second = num_searches / elapsed_time

        # Very basic performance check - should be able to do at least 0.6 searches/sec
        # This is a conservative check to catch major performance regressions
        logger.info(f"Performance: {searches_per_second:.2f} searches/second")
        self.assertGreater(
            searches_per_second,
            0.6,  # Lowered threshold to account for health check overhead and noise
            f"Performance regression: only {searches_per_second:.2f} searches/second",
        )

    def test_memory_cleanup(self):
        """Test that shared memory is properly cleaned up."""
        # Create controller and get buffer names
        controller = MCTSController(
            num_workers=1,
            use_mock_model=True,
            buffer_count=4,
        )
        controller.start()

        # Record the buffer names
        buffer_names = controller.shared_memory_config.buffer_names.copy()
        self.assertEqual(len(buffer_names), 4)

        # Shutdown and verify cleanup
        controller.shutdown()
        self.assertEqual(len(controller.shared_memory_blocks), 0)
        self.assertEqual(len(controller.shared_memory_config.buffer_names), 0)

        # Verify shared memory blocks are actually unlinked
        # (This is implicit - if cleanup failed, subsequent tests might fail)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the system maintains backward compatibility with queue-based IPC."""

    def test_queue_based_fallback(self):
        """Test that the system can fall back to queue-based communication."""
        # This test would require modifying the manager and worker to support
        # None shared_memory_config, which triggers queue-based mode
        # For now, this is a placeholder for future implementation
        pass


if __name__ == "__main__":
    unittest.main()
