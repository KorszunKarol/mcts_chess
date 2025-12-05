"""
Extract Keras model outputs for comparison.
Run this in the 'tf' environment.
"""

import sys
import os
import numpy as np
import chess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder
import tensorflow as tf


def load_keras_model(model_path: str):
    """Load the Keras model with swish activation."""
    custom_objects = {"swish": tf.nn.silu}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    return model


def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a chess position."""
    encoder = Encoder()
    return encoder.encode(board)


def keras_inference(model, position: np.ndarray):
    """Run inference with Keras model."""
    input_tensor = np.expand_dims(position, axis=0)
    outputs = model.predict(input_tensor, verbose=0)
    value = outputs[0][0]
    policy = outputs[1][0]
    return value, policy


def main():
    model_path = "src/weights/best_model.keras"
    output_file = "saved_models/keras_outputs.json"
    
    print("Loading Keras model...")
    model = load_keras_model(model_path)
    print("✓ Model loaded")
    
    # Test positions
    test_positions = [
        ("starting_position", chess.Board()),
        ("after_e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
        ("ruy_lopez", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
        ("middle_game", chess.Board("r2qkb1r/pp2nppp/2n1p3/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq - 0 6")),
        ("endgame", chess.Board("8/5k2/8/8/8/5K2/8/8 w - - 0 1")),
    ]
    
    encoder = Encoder()
    results = {}
    
    for name, board in test_positions:
        print(f"\nProcessing: {name}")
        position = encoder.encode(board)
        value, policy = keras_inference(model, position)
        
        results[name] = {
            "value": value.tolist(),
            "policy": policy.tolist(),
        }
        print(f"  Value: {value}")
        print(f"  Policy top-5: {np.argsort(policy)[-5:][::-1]}")
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved outputs to: {output_file}")


if __name__ == '__main__':
    main()

