"""
Compare PyTorch outputs with pre-extracted Keras outputs.
Run this in the 'chess' environment.
"""

import sys
import os
import numpy as np
import chess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder
import torch
from src.transformer_model_pytorch import create_model


def load_pytorch_model(model_path: str):
    """Load the PyTorch model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a chess position."""
    encoder = Encoder()
    return encoder.encode(board)


def pytorch_inference(model, position: np.ndarray):
    """Run inference with PyTorch model."""
    # Convert from (8, 8, 34) to (34, 8, 8) - NCHW format
    # Make a copy to avoid negative stride issues
    input_tensor = np.transpose(position.copy(), (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = torch.from_numpy(input_tensor).float()
    
    model.eval()
    with torch.no_grad():
        value, policy = model(input_tensor)
    
    value = value[0].numpy()
    policy = policy[0].numpy()
    return value, policy


def compare_outputs(keras_value, keras_policy, pytorch_value, pytorch_policy, tolerance=1e-3):
    """Compare outputs and report differences."""
    # Value comparison
    value_diff = np.abs(keras_value - pytorch_value)
    value_max_diff = np.max(value_diff)
    value_mean_diff = np.mean(value_diff)
    
    print(f"  Value - Keras:   {keras_value}")
    print(f"  Value - PyTorch: {pytorch_value}")
    print(f"  Value - Max diff:  {value_max_diff:.6f}, Mean diff: {value_mean_diff:.6f}")
    
    # Policy comparison
    policy_diff = np.abs(keras_policy - pytorch_policy)
    policy_max_diff = np.max(policy_diff)
    policy_mean_diff = np.mean(policy_diff)
    
    keras_top5 = np.argsort(keras_policy)[-5:][::-1]
    pytorch_top5 = np.argsort(pytorch_policy)[-5:][::-1]
    
    print(f"  Policy - Max diff:  {policy_max_diff:.6f}, Mean diff: {policy_mean_diff:.6f}")
    print(f"  Policy - Top 5 (Keras):   {keras_top5}")
    print(f"  Policy - Top 5 (PyTorch): {pytorch_top5}")
    print(f"  Policy - Top 5 overlap: {len(set(keras_top5) & set(pytorch_top5))}/5")
    
    value_match = value_max_diff < tolerance
    policy_match = policy_max_diff < tolerance
    top5_match = np.array_equal(keras_top5, pytorch_top5)
    
    if value_match and policy_match:
        print(f"  ✓ Outputs match (within {tolerance})")
        if top5_match:
            print(f"  ✓ Top 5 moves match!")
    else:
        print(f"  ✗ Outputs differ")
    
    return value_match and policy_match


def main():
    keras_outputs_file = "saved_models/keras_outputs.json"
    pytorch_model_path = "saved_models/best_model_pytorch.pt"
    
    print("="*70)
    print("KERAS vs PYTORCH MODEL COMPARISON")
    print("="*70)
    
    # Load Keras outputs
    print(f"\nLoading Keras outputs from: {keras_outputs_file}")
    with open(keras_outputs_file, 'r') as f:
        keras_outputs = json.load(f)
    print("✓ Keras outputs loaded")
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model from: {pytorch_model_path}")
    pytorch_model = load_pytorch_model(pytorch_model_path)
    print("✓ PyTorch model loaded")
    
    # Test positions (same as in extract_keras_outputs.py)
    test_positions = [
        ("starting_position", chess.Board()),
        ("after_e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
        ("ruy_lopez", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
        ("middle_game", chess.Board("r2qkb1r/pp2nppp/2n1p3/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq - 0 6")),
        ("endgame", chess.Board("8/5k2/8/8/8/5K2/8/8 w - - 0 1")),
    ]
    
    encoder = Encoder()
    all_match = True
    
    for name, board in test_positions:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        # Get Keras outputs
        keras_data = keras_outputs[name]
        keras_value = np.array(keras_data["value"])
        keras_policy = np.array(keras_data["policy"])
        
        # Run PyTorch inference
        position = encoder.encode(board)
        pytorch_value, pytorch_policy = pytorch_inference(pytorch_model, position)
        
        # Compare
        match = compare_outputs(keras_value, keras_policy, pytorch_value, pytorch_policy, tolerance=1e-3)
        all_match = all_match and match
    
    print("\n" + "="*70)
    if all_match:
        print("✓ CONVERSION SUCCESSFUL: All positions match!")
    else:
        print("⚠ CONVERSION WARNING: Some positions differ")
        print("  This may be due to numerical precision or implementation differences")
    print("="*70 + "\n")
    
    return all_match


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

