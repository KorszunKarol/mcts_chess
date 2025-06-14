import argparse
import os
import tensorflow as tf
from datetime import datetime

from src.model import create_model
from src.data.dataset import create_dataset
from src.training.losses import create_combined_loss


def train(
    pgn_path: str,
    weights_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    min_elo: int,
    loss_alpha: float,
):
    """
    Main training function to bootstrap the model from PGN data.
    """
    print("--- Starting Training ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Physical GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # 1. Create the dataset pipeline
    print(f"Loading data from: {pgn_path}")
    dataset = create_dataset(pgn_path=pgn_path, batch_size=batch_size, min_elo=min_elo)
    print("Dataset created successfully.")

    # 2. Create and compile the model
    print("Creating and compiling the model...")
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = create_combined_loss(alpha=loss_alpha)

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.summary()

    if weights_path and os.path.exists(weights_path):
        print(f"Loading initial weights from {weights_path}")
        model.load_weights(weights_path)

    # 3. Set up Keras callbacks for monitoring
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # Checkpoint to save the best model based on validation loss (if we add a val set)
    # For now, we save based on training loss.
    checkpoint_path = "weights/bootstrap_model.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_path}")

    # 4. Run the training loop
    print("Starting model.fit()...")
    model.fit(
        dataset, epochs=epochs, callbacks=[tensorboard_callback, checkpoint_callback]
    )
    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the chess model from PGN data.")

    parser.add_argument(
        "--pgn", type=str, required=True, help="Path to the PGN file for training."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to load initial model weights from (optional).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--min_elo",
        type=int,
        default=2400,
        help="Minimum ELO of players for games to be included.",
    )
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for the value loss (0.0 to 1.0).",
    )

    args = parser.parse_args()

    train(
        pgn_path=args.pgn,
        weights_path=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_elo=args.min_elo,
        loss_alpha=args.loss_alpha,
    )
