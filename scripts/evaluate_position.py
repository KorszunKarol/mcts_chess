# ==============================================================================
# scripts/evaluate_position.py
# ==============================================================================
# This script loads two models and evaluates a single, hard-coded FEN position,
# printing a side-by-side comparison of their value and policy outputs.

import sys
import os
import chess
import numpy as np
import tensorflow as tf

# --- Robust path setup for module imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import model creation functions with aliases to avoid name collision
from src.model import create_model as create_cnn_model
from src.transformer_model import create_model as create_transformer_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move

# ============================ CONFIGURATION ===================================
# --- Path to your trained Keras model files ---
CNN_MODEL_PATH = os.path.join(project_root, 'src/weights/best_model_cnn.keras')
TRANSFORMER_MODEL_PATH = os.path.join(project_root, 'src/weights/best_model.keras')

# --- FEN to Evaluate ---
# Change this FEN string to evaluate any position you want.
FEN_TO_EVALUATE = "rnbq3r/pp1kbQ1p/8/2ppPpB1/8/2PP3P/P1P3P1/R3KBNR b KQ - 2 11"
# ==============================================================================

class ModelEvaluator:
    """A helper class to load a model and evaluate a position."""
    def __init__(self, model_path: str, model_creation_fn, model_name: str):
        self.model_name = model_name
        print(f"Loading {self.model_name}...")
        self.model = model_creation_fn()
        try:
            self.model.load_weights(model_path)
            self.encoder = Encoder()
            print(f"{self.model_name} loaded.")
        except Exception as e:
            print(f"FATAL: Could not load {self.model_name}. Error: {e}")
            raise

    def evaluate_position(self, board: chess.Board):
        """Prints the model's evaluation of the given board state."""
        print(f"\n--- Evaluation by: {self.model_name} ---")

        # Encode the board and predict
        encoded_state = self.encoder.encode(board)
        input_tensor = np.expand_dims(encoded_state, axis=0)
        value_pred, policy_logits = self.model.predict(input_tensor, verbose=0)

        value_pred = value_pred[0]
        policy_logits = policy_logits[0]

        # --- Print Value Head Output ---
        print(f"Value Prediction: [Win: {value_pred[0]:.2%}, Loss: {value_pred[1]:.2%}, Draw: {value_pred[2]:.2%}]")

        # --- Print Policy Head Output ---
        legal_moves = list(board.legal_moves)
        legal_move_indices = {move_to_index(move, board) for move in legal_moves}

        # Get probabilities for legal moves only
        legal_logits = np.array([policy_logits[i] for i in legal_move_indices if i is not None])
        if not legal_logits.any():
            print("No legal moves with positive logits found.")
            return

        legal_probabilities = tf.nn.softmax(legal_logits).numpy()

        # Pair legal moves with their probabilities and sort
        move_probabilities = sorted(
            zip([move for move in legal_moves if move_to_index(move, board) is not None], legal_probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        print("Policy Predictions (Top 5):")
        for move, prob in move_probabilities[:5]:
            print(f"  - Move: {board.san(move):<8} | Probability: {prob:.3f}")
        print("-" * (20 + len(self.model_name)))

def main():
    """Main function to run the evaluation."""
    # --- GPU Configuration ---
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

    # --- Load Models ---
    try:
        cnn_evaluator = ModelEvaluator(CNN_MODEL_PATH, create_cnn_model, "CNN Model")
        transformer_evaluator = ModelEvaluator(TRANSFORMER_MODEL_PATH, create_transformer_model, "Transformer Model")
    except Exception:
        print("Aborting due to model loading failure.")
        return

    # --- Evaluate Position ---
    board = chess.Board(FEN_TO_EVALUATE)
    print("\n" + "="*50)
    print(f"Evaluating Position: {board.fen()}")
    print(board)
    print("="*50)

    cnn_evaluator.evaluate_position(board)
    transformer_evaluator.evaluate_position(board)
    print()


if __name__ == '__main__':
    if not os.path.exists(CNN_MODEL_PATH) or not os.path.exists(TRANSFORMER_MODEL_PATH):
        print("FATAL: One or both model files not found. Please check paths in configuration.")
    else:
        main()