# ==============================================================================
# scripts/run_mcts_search.py
# ==============================================================================
# This script demonstrates how to use the MCTSController to get a policy
# prediction for a given board position.

import sys
import os
import chess
import logging
from datetime import datetime

# --- Robust path setup for module imports ---
# Go up two directory levels to the project root (e.g., chess_2.0/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.mcts.controller import MCTSController

# ============================ CONFIGURATION ===================================
# --- Path to your BEST trained Keras model file ---
# IMPORTANT: Update this path to the model you want to use for evaluation.
# You can use the CNN model, the Transformer model, etc.
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'src/weights/best_model.keras')

# --- MCTS Search Parameters ---
# The board position to analyze, in FEN notation.
FEN_TO_ANALYZE = "4rk2/3n3p/1p1q2pP/p2p1pP1/3P1P2/1P5B/P2Q1K2/2R5 b - - 0 31"  # Starting position

# Number of simulations to run. More simulations lead to a stronger evaluation
# but take more time. A value between 800 and 1600 is typical for strong play.
NUM_SIMULATIONS = 10_000

# --- MODIFICATION: Add batch size configuration ---
# The number of evaluations to batch together on the GPU.
# Smaller batches use less VRAM. Start with a safe value like 16 or 32.
MCTS_BATCH_SIZE = 256
# --- END OF MODIFICATION ---

# --- Engine Configuration ---
# Number of CPU processes to use for the search.
NUM_WORKERS = 8
# ==============================================================================

class StreamToLogger:
    """
    A file-like stream object that redirects writes to a logger instance.
    This is used to capture all stdout/stderr output.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # The buffer can contain multiple lines
        for line in buf.rstrip().splitlines():
            # Pass the line to the logger
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # This is needed for the file-like interface.
        pass

def setup_logging():
    """
    Configures logging to capture all stdout/stderr to a file and also
    print to the original console.
    A new log file is created in the 'logs/' directory for each run.
    """
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"mcts_search_run_{timestamp}.log")

    # Keep original streams to prevent infinite recursion
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File Handler - writes all logs to the file
    file_handler = logging.FileHandler(log_filename)
    # The file log should be comprehensive
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler - writes to the original stdout for interactive display
    console_handler = logging.StreamHandler(original_stdout)
    console_formatter = logging.Formatter('%(message)s') # Keep it clean
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Redirect stdout and stderr to the logging system
    # All `print` statements and other direct stdout/stderr writes will now
    # be captured by our logger.
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

    logging.info(f"--- Logging configured. All console output will be saved to {log_filename} ---")
    return log_filename

def analyze_position():
    """
    Sets up the MCTS controller, runs a search for the specified position,
    and logs the resulting policy.
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logging.error(f"FATAL: Model weights file not found at '{MODEL_WEIGHTS_PATH}'.")
        logging.error("Please update the MODEL_WEIGHTS_PATH in this script.")
        return

    # The MCTSController is best used as a context manager to ensure that all
    # background processes and shared memory are cleaned up correctly.
    try:
        with MCTSController(
            num_workers=NUM_WORKERS,
            model_weights_path=MODEL_WEIGHTS_PATH,
            batch_size=MCTS_BATCH_SIZE,
            use_mock_model=False,
            max_wait_time_ms=50.0
        ) as controller:

            logging.info(f"Starting MCTS search for FEN: {FEN_TO_ANALYZE}")
            logging.info(f"Running {NUM_SIMULATIONS} simulations with {NUM_WORKERS} workers...")

            # --- Run the search ---
            result = controller.run_search(FEN_TO_ANALYZE, num_simulations=NUM_SIMULATIONS)

            # --- Print the results ---
            if result.error:
                logging.error(f"\nAn error occurred during search: {result.error}")
            else:
                logging.info("\n--- Search Complete ---")
                logging.info(f"Final evaluation (Q-value): {result.q_value:.4f}")
                logging.info("Policy (Top 10 moves):")

                # Sort the policy dictionary by probability (visit count)
                sorted_policy = sorted(result.policy.items(), key=lambda item: item[1], reverse=True)

                # Pretty-print the top 10 moves and their probabilities
                board = chess.Board(FEN_TO_ANALYZE)
                for i, (move, probability) in enumerate(sorted_policy[:10]):
                    san_move = board.san(move)
                    logging.info(f"  {i+1}. {san_move:<8} | Probability: {probability:.4f}")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    # It's good practice to set the multiprocessing start method to 'spawn'
    # at the beginning of your script, especially for CUDA compatibility.
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn")
    except RuntimeError:
        # The start method can only be set once.
        pass

    log_filename = setup_logging()
    # This initial message will be logged and printed, confirming setup.
    logging.info(f"To view the log in real-time, you can run:\ntail -f {log_filename}\n")

    # Measure execution time
    import time
    start_time = time.time()

    analyze_position()

    # Calculate and log the execution time
    execution_time = time.time() - start_time
    logging.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")