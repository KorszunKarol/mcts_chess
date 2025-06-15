import os
import tensorflow as tf
from src.data.dataset import create_dataset
from src.model import create_model

# --- Configuration Constants ---
PGN_PATH = "data/processed/lichess_db_standard_rated_2016-01.pgn"
OUTPUT_DIR = "training_output"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
# -----------------------------

def main():
    """
    Main training routine.
    """
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # Component Initialization
    try:
        train_dataset = create_dataset_from_folder(PGN_PATH, BATCH_SIZE)
    except FileNotFoundError:
        print(f"Error: PGN file not found at {PGN_PATH}")
        return

    model = create_model()

    # Model Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss={
            'value_head': tf.keras.losses.MeanSquaredError(),
            'policy_head': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={'value_head': 0.5, 'policy_head': 0.5},
        metrics={
            'value_head': 'mae',
            'policy_head': 'accuracy'
        }
    )

    model.summary()

    # Callback Configuration
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'models', 'best_model.h5'),
        save_weights_only=False,
        monitor='loss', # Monitoring training loss as there is no validation split mentioned
        mode='min',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(OUTPUT_DIR, 'logs'),
        histogram_freq=1)

    callbacks = [model_checkpoint_callback, tensorboard_callback]

    # Execute Training
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

if __name__ == '__main__':
    # Create output directories if they don't exist
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)

    main()