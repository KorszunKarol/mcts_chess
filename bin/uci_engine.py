# ==============================================================================
# uci_engine.py
# ==============================================================================

# --- Pre-import Sanity Check ---
# This is a raw file write to debug startup. It uses no modules.
with open("startup_check.log", "w") as f:
    f.write("uci_engine.py script started execution.\n")

# This script acts as a UCI-compatible engine, allowing you to play against
# your raw trained model (without MCTS) in a standard chess GUI.

import sys
import os
import chess
import numpy as np
import tensorflow as tf
import logging

# --- Robust path setup for module imports ---
# This ensures 'src' is found, making the script runnable from bin/ directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # bin/ is one level down from root
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import create_model
from src.transformer_model import create_model as create_transformer_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move

# ============================ CONFIGURATION ===================================
# --- Path to your BEST trained Keras model file ---
MODEL_PATH = '/home/karolito/DL/chess_2.0/src/weights/best_model.keras'
LOG_FILE = 'uci_engine.log'
# ==============================================================================

# --- Set up Logging ---
# We log to a file because UCI communication happens over stdin/stdout.
# Printing to the console for debugging would break the UCI protocol.
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(message)s', filemode='w')

class UCIEngine:
    """
    The main class to handle UCI communication and run the chess engine.
    """
    def __init__(self, model_path):
        logging.info("Initializing UCI Engine...")
        self.board = chess.Board()
        try:
            # Suppress verbose TensorFlow output on load
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            self.model = create_transformer_model()
            self.model.load_weights(model_path)
            self.encoder = Encoder()
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"FATAL: Could not load model. Error: {e}")
            raise

    def get_ai_move(self):
        """Gets the best move from the raw policy head of the neural network."""
        logging.info(f"Board FEN for prediction: {self.board.fen()}")
        encoded_state = self.encoder.encode(self.board)
        input_tensor = np.expand_dims(encoded_state, axis=0)

        # Suppress prediction progress bars in the log
        value_pred, policy_logits = self.model.predict(input_tensor, verbose=0)
        policy_logits = policy_logits[0]

        legal_move_indices = {move_to_index(move, self.board) for move in self.board.legal_moves}

        # Create a mask where legal moves have a large positive value (0) and illegal moves have a large negative value.
        masked_policy = np.full_like(policy_logits, -1e9)
        for index in legal_move_indices:
            if index is not None:
                masked_policy[index] = policy_logits[index]

        # --- Enhanced Logging for Debugging ---
        # Apply softmax to the masked logits to get probabilities for legal moves.
        # We only consider the logits for legal moves to create a proper probability distribution.
        legal_logits = np.array([policy_logits[i] for i in legal_move_indices if i is not None])
        legal_probabilities = tf.nn.softmax(legal_logits).numpy()

        # Pair legal moves with their probabilities
        move_probabilities = sorted(
            zip([index_to_move(i) for i in legal_move_indices if i is not None], legal_probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        logging.info("--- Top 5 Moves & Probabilities ---")
        for move, prob in move_probabilities[:5]:
            logging.info(f"Move: {move.uci():<6} Probability: {prob:.4f}")
        logging.info("------------------------------------")

        best_move_index = np.argmax(masked_policy)
        best_move = index_to_move(best_move_index)

        # This check is a safeguard. If the chosen move is somehow illegal, fall back to any legal move.
        if best_move not in self.board.legal_moves:
            logging.warning(f"NN chose an illegal move: {best_move}. Falling back to a random legal move.")
            best_move = list(self.board.legal_moves)[0]

        value_wld = value_pred[0]
        logging.info(f"AI Evaluation: Win: {value_wld[0]:.2%}, Loss: {value_wld[1]:.2%}, Draw: {value_wld[2]:.2%}")
        return best_move

    def handle_position(self, line: str):
        """Parses the 'position' command and updates the board state."""
        parts = line.split()
        moves_start_index = -1

        if len(parts) > 1 and parts[1] == 'startpos':
            self.board.reset()
            if len(parts) > 2 and parts[2] == 'moves':
                moves_start_index = 3
        elif len(parts) > 1 and parts[1] == 'fen':
            try:
                # The FEN string can be up to 6 parts
                fen = " ".join(parts[2:8])
                self.board.set_fen(fen)
                if len(parts) > 8 and parts[8] == 'moves':
                    moves_start_index = 9
            except (ValueError, IndexError):
                 logging.error(f"Invalid FEN string received: {line}")
                 return
        else:
            logging.warning(f"Could not parse 'position' command: {line}")
            return

        if moves_start_index != -1:
            for move_uci in parts[moves_start_index:]:
                try:
                    self.board.push_uci(move_uci)
                except ValueError:
                    logging.error(f"Invalid move received from GUI: {move_uci}")

        logging.info(f"Board position set to: {self.board.fen()}")

    def handle_go(self):
        """Parses the 'go' command and starts the search."""
        logging.info("Received 'go' command. Getting AI move...")
        best_move = self.get_ai_move()
        logging.info(f"Search complete. Best move found: {best_move.uci()}")
        print(f"bestmove {best_move.uci()}", flush=True)

    def uci_loop(self):
        """The main loop to listen for and respond to UCI commands."""
        logging.info("UCI Engine main loop started. Waiting for commands.")

        # Ensure stdout is not buffered
        sys.stdout.reconfigure(line_buffering=True)

        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            logging.info(f"GUI -> Engine: {line}")

            if line == "uci":
                print("id name My-Keras-Engine")
                print("id author Karolito")
                print("uciok")
            elif line == "isready":
                print("readyok")
            elif line.startswith("position"):
                self.handle_position(line)
            elif line.startswith("go"):
                self.handle_go()
            elif line == "quit":
                logging.info("Quit command received. Exiting.")
                break
            # Other commands like 'ucinewgame' can be ignored for a simple engine

if __name__ == '__main__':
    # It's better to configure GPU memory growth at the start of the script
    # to avoid potential issues with TensorFlow initialization.
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("GPU memory growth enabled.")
    except RuntimeError as e:
        logging.error(f"GPU memory growth error: {e}")

    if not os.path.exists(MODEL_PATH):
        logging.critical(f"FATAL: Model file not found at '{MODEL_PATH}'. Engine cannot start.")
    else:
        try:
            engine = UCIEngine(MODEL_PATH)
            engine.uci_loop()
        except Exception as e:
            logging.critical(f"A critical error occurred during engine initialization or main loop: {e}", exc_info=True)