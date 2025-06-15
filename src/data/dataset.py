# src/data/dataset.py
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from typing import Generator, Tuple

from src.encoder import Encoder
from src.move_mapping import move_to_index, ACTION_SPACE_SIZE
from src.model import create_model  # To get shapes from the model definition itself

# Get model-specific shapes
_temp_model = create_model()
STATE_SHAPE = _temp_model.input_shape[1:]
POLICY_SHAPE = (_temp_model.get_layer("policy_head").output.shape[1],)
del _temp_model




def pgn_data_generator(
    pgn_path: str, min_elo: int, encoder: Encoder
) -> Generator[Tuple[np.ndarray, int, np.ndarray], None, None]:
    """
    A Python generator that reads a PGN file, filters games, and yields training samples.

    This function is the core of the data pipeline. It reads one game at a time
    to keep memory usage low, making it suitable for very large PGN files.

    Args:
        pgn_path: The full path to the PGN file.
        min_elo: The minimum ELO rating for both players. Games below this are skipped.
        encoder: An instance of the Encoder class to transform board states.

    Yields:
        A tuple of (state_tensor, policy_target, value_target) for each move
        in a valid game.
    """
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn_file:
        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
            except (ValueError, IndexError) as e:
                # Corrupted game data, skip to the next
                print(f"Skipping corrupted game: {e}")
                continue

            if game is None:
                break  # End of file

            # FIX: Check for parsing errors recorded by the python-chess library.
            # If a game's movetext is malformed, it's not trustworthy.
            if game.errors:
                # print(f"Skipping game with parsing errors: {game.errors}")
                continue

            # --- 1. Header Validation ---
            try:
                white_elo = int(game.headers["WhiteElo"])
                black_elo = int(game.headers["BlackElo"])
                termination = game.headers["Termination"]
                result_str = game.headers["Result"]

                if white_elo < min_elo or black_elo < min_elo:
                    continue
                if termination in ["Time forfeit", "Abandoned"]:
                    continue
                if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                    continue

            except (KeyError, ValueError):
                continue  # Skip game if required headers are missing or invalid

            # --- 2. Outcome Transformation ---
            if result_str == "1-0":
                game_outcome = 1.0
            elif result_str == "0-1":
                game_outcome = -1.0
            else:  # "1/2-1/2"
                game_outcome = 0.0

            # --- 3. Mainline Move Iteration ---
            board = game.board()
            try:
                for move in game.mainline_moves():
                    value_target = (
                        game_outcome if board.turn == chess.WHITE else -game_outcome
                    )
                    state_tensor = encoder.encode(board)
                    policy_idx = move_to_index(move, board)
                    if policy_idx is None:
                        # This can happen if the move is exotic and not in our mapping.
                        # We treat this as a corruption for this game.
                        raise ValueError(f"Move {move} not in action space or illegal.")

                    policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                    policy_target[policy_idx] = 1.0

                    yield (state_tensor, policy_target, value_target)
                    board.push(move)
            except Exception:
                # If anything goes wrong during move processing, skip the game.
                continue


def create_dataset(
    pgn_path: str,
    batch_size: int,
    min_elo: int = 2400,
    shuffle_buffer_size: int = 100_000,
) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset pipeline for training.
    """
    encoder = Encoder()

    output_signature = (
        tf.TensorSpec(shape=STATE_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=POLICY_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: pgn_data_generator(pgn_path, min_elo, encoder),
        output_signature=output_signature,
    )

    dataset = dataset.map(lambda state, policy, value: (state, (value, policy)))

    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
