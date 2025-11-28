# ==============================================================================
# play_vs_ai.py
# ==============================================================================
# Play a game of chess against your raw, trained neural network without MCTS.

import os
import chess
import chess.svg
import numpy as np
import tensorflow as tf
import sys

# --- Robust path setup for module imports ---
# This ensures that the 'src' directory is found regardless of where the script is executed.
# It adds the project's root directory ('chess_2.0') to the Python path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import create_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move

# ============================ CONFIGURATION ===================================
# --- Path to your BEST trained Keras model file ---
MODEL_PATH = '/home/karolito/DL/chess_2.0/src/weights/model_20250616-003116_epoch_30.keras'
# ==============================================================================


class ChessPlayer:
    def __init__(self, model_path):
        print("Loading model...")
        self.model = create_model()
        self.model.load_weights(model_path)
        self.encoder = Encoder()
        print("Model loaded successfully.")

    def get_ai_move(self, board: chess.Board):
        """
        Gets the best move from the raw policy head of the neural network.
        """
        # 1. Encode the current board state
        encoded_state = self.encoder.encode(board)

        # 2. Add a batch dimension and predict
        # The model expects a batch, so we add a dimension: (8, 8, 34) -> (1, 8, 8, 34)
        input_tensor = np.expand_dims(encoded_state, axis=0)
        value_pred, policy_logits = self.model.predict(input_tensor, verbose=0)

        policy_logits = policy_logits[0] # Remove the batch dimension

        # 3. Mask out illegal moves
        # This is a critical step! The network might assign probabilities to illegal moves.
        # We must filter them out before choosing the best one.
        legal_move_indices = {move_to_index(move, board) for move in board.legal_moves}

        # Create a mask where legal moves are 1 and illegal moves are 0
        legal_mask = np.zeros_like(policy_logits)
        for index in legal_move_indices:
            if index is not None:
                legal_mask[index] = 1

        # Apply the mask. We add a large negative number to illegal moves so they are
        # never chosen by the softmax or argmax operations.
        masked_policy = policy_logits + (legal_mask - 1) * 1e9

        # 4. Find the best legal move
        # The index of the highest logit corresponds to the best move.
        best_move_index = np.argmax(masked_policy)
        best_move = index_to_move(best_move_index)

        # Print the model's evaluation
        value_win_loss_draw = value_pred[0]
        print(f"AI Evaluation: Win: {value_win_loss_draw[0]:.2%}, Loss: {value_win_loss_draw[1]:.2%}, Draw: {value_win_loss_draw[2]:.2%}")

        return best_move

def play_game():
    """Main game loop for a human vs. AI game."""
    board = chess.Board()
    player = ChessPlayer(MODEL_PATH)

    # Decide who plays what color
    human_color = None
    choice = None
    while choice not in ['white', 'black']:
        choice = input("Do you want to play as 'white' or 'black'? ").lower()
        if choice == 'white':
            human_color = chess.WHITE
        elif choice == 'black':
            human_color = chess.BLACK

    while not board.is_game_over():
        # Print the board from the perspective of the current player
        print("\n" + "="*20)
        print(board)
        print("="*20)

        if board.turn == human_color:
            # Human's turn
            move_uci = ""
            while True:
                try:
                    move_uci = input("Enter your move in UCI format (e.g., e2e4): ")
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Invalid or illegal move. Try again.")
                except ValueError:
                    print("Invalid move format. Please use UCI (e.g., e2e4).")
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = player.get_ai_move(board)
            print(f"AI plays: {ai_move.uci()}")
            board.push(ai_move)

    # Print the final result
    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")
    print(board)


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure you have a trained 'best_model.keras' in the correct directory.")
    else:
        play_game()