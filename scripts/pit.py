# ==============================================================================
# scripts/pit.py
# ==============================================================================
# This script pits two trained models against each other from a specified FEN.

import sys
import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from datetime import datetime

# --- Robust path setup for module imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CORRECTLY IMPORT MODELS WITH ALIASES TO AVOID NAME COLLISION ---
from src.model import create_model as create_cnn_model
from src.transformer_model import create_model as create_transformer_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move

# ============================ CONFIGURATION ===================================
# --- Paths to your trained Keras model files ---
WHITE_MODEL_PATH = os.path.join(project_root, 'src/weights/best_model_cnn.keras')
BLACK_MODEL_PATH = os.path.join(project_root, 'src/weights/best_model.keras')

# --- Starting position ---
STARTING_FEN = "rnbqkb1r/pp4pp/8/2ppPp2/8/2PP1Q2/P1P3PP/R1B1KBNR w KQkq - 0 8"
# ==============================================================================

class ChessAIPlayer:
    """A player that uses a trained Keras model to select moves."""
    def __init__(self, model_path: str, player_name: str, model_creation_fn):
        print(f"Loading model for {player_name} from {os.path.basename(model_path)}...")
        self.model = model_creation_fn()
        try:
            self.model.load_weights(model_path)
            self.encoder = Encoder()
            self.name = player_name
            print(f"Model for {player_name} loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load model from {model_path}. Error: {e}")
            raise

    def get_move(self, board: chess.Board):
        """Gets the best move from the raw policy head of the neural network."""
        encoded_state = self.encoder.encode(board)
        input_tensor = np.expand_dims(encoded_state, axis=0)
        value_pred, policy_logits = self.model.predict(input_tensor, verbose=0)
        policy_logits = policy_logits[0]

        legal_move_indices = {move_to_index(move, board) for move in board.legal_moves}

        masked_policy = np.full_like(policy_logits, -np.inf)
        for index in legal_move_indices:
            if index is not None:
                masked_policy[index] = policy_logits[index]

        best_move_index = np.argmax(masked_policy)
        best_move = index_to_move(best_move_index)

        # Safeguard: if the best move is somehow illegal, choose a random legal move
        if best_move not in board.legal_moves:
            print(f"WARNING: {self.name} chose an illegal move ({best_move}). Choosing random move.")
            return list(board.legal_moves)[0]

        return best_move

def pit_models():
    """Main game loop for a model vs. model game."""
    # --- GPU Configuration ---
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

    # --- Load Players ---
    try:
        white_player = ChessAIPlayer(WHITE_MODEL_PATH, "CNN (White)", create_cnn_model)
        black_player = ChessAIPlayer(BLACK_MODEL_PATH, "Transformer (Black)", create_transformer_model)
    except Exception:
        print("Aborting due to model loading failure.")
        return

    # --- Setup Game ---
    board = chess.Board(STARTING_FEN)
    game = chess.pgn.Game()
    game.headers["Event"] = "Model Pit"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = white_player.name
    game.headers["Black"] = black_player.name
    game.setup(board)
    node = game

    print("\n--- Game Start ---")
    print(f"Starting Position (FEN): {board.fen()}")

    # --- Game Loop ---
    while not board.is_game_over():
        print("\n" + "="*40)
        print(board)
        print("="*40)

        if board.turn == chess.WHITE:
            player = white_player
            move_number_str = f"{board.fullmove_number}."
        else:
            player = black_player
            move_number_str = f"{board.fullmove_number}... "


        print(f"Thinking for {player.name}...")
        ai_move = player.get_move(board)
        print(f"{move_number_str} {player.name} plays: {board.san(ai_move)}")
        board.push(ai_move)
        node = node.add_variation(ai_move)

    # --- Game Over ---
    print("\n" + "="*40)
    print("--- GAME OVER ---")
    print(f"Final Position:\n{board}")
    print(f"Result: {board.result()}")

    # --- Save PGN ---
    game.headers["Result"] = board.result()
    pgn_filename = f"pit_{white_player.name.split(' ')[0]}_vs_{black_player.name.split(' ')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    games_dir = os.path.join(project_root, 'games')
    os.makedirs(games_dir, exist_ok=True)
    pgn_path = os.path.join(games_dir, pgn_filename)

    with open(pgn_path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

    print(f"\nGame saved to PGN: {pgn_path}")
    print("="*40)


if __name__ == '__main__':
    if not os.path.exists(WHITE_MODEL_PATH) or not os.path.exists(BLACK_MODEL_PATH):
        print("FATAL: One or both model files not found. Please check the paths in the configuration.")
    else:
        pit_models()