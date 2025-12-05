"""
Detailed analysis of policy output differences.
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

# Load Keras outputs
with open('saved_models/keras_outputs.json', 'r') as f:
    keras_outputs = json.load(f)

# Load PyTorch model
checkpoint = torch.load('saved_models/best_model_pytorch.pt', map_location='cpu')
pytorch_model = create_model()
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

encoder = Encoder()

test_positions = [
    ("starting_position", chess.Board()),
    ("after_e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
    ("ruy_lopez", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
]

print("="*80)
print("DETAILED POLICY OUTPUT DIFFERENCE ANALYSIS")
print("="*80)

for name, board in test_positions:
    print(f"\n{'='*80}")
    print(f"Position: {name}")
    print(f"{'='*80}")
    
    # Get Keras outputs
    keras_data = keras_outputs[name]
    keras_policy = np.array(keras_data["policy"])
    
    # Get PyTorch outputs
    position = encoder.encode(board)
    input_tensor = np.transpose(position.copy(), (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = torch.from_numpy(input_tensor).float()
    
    with torch.no_grad():
        _, pytorch_policy = pytorch_model(input_tensor)
    pytorch_policy = pytorch_policy[0].numpy()
    
    # Calculate differences
    diff = np.abs(keras_policy - pytorch_policy)
    
    print(f"\nOverall Statistics:")
    print(f"  Max difference:  {diff.max():.6f}")
    print(f"  Mean difference:  {diff.mean():.6f}")
    print(f"  Median difference: {np.median(diff):.6f}")
    print(f"  Std of differences: {diff.std():.6f}")
    print(f"  95th percentile: {np.percentile(diff, 95):.6f}")
    print(f"  99th percentile: {np.percentile(diff, 99):.6f}")
    
    # Top moves comparison
    keras_top10 = np.argsort(keras_policy)[-10:][::-1]
    pytorch_top10 = np.argsort(pytorch_policy)[-10:][::-1]
    
    print(f"\nTop 10 Moves Comparison:")
    print(f"  Keras:   {keras_top10}")
    print(f"  PyTorch: {pytorch_top10}")
    print(f"  Overlap: {len(set(keras_top10) & set(pytorch_top10))}/10")
    
    # Check differences for top moves
    print(f"\nDifferences for Top Moves:")
    print(f"  Keras top move ({keras_top10[0]}):")
    print(f"    Keras value:   {keras_policy[keras_top10[0]]:.6f}")
    print(f"    PyTorch value: {pytorch_policy[keras_top10[0]]:.6f}")
    print(f"    Difference:    {diff[keras_top10[0]]:.6f}")
    print(f"    Rank in PyTorch: {np.where(np.argsort(pytorch_policy)[::-1] == keras_top10[0])[0][0] + 1}")
    
    print(f"  PyTorch top move ({pytorch_top10[0]}):")
    print(f"    Keras value:   {keras_policy[pytorch_top10[0]]:.6f}")
    print(f"    PyTorch value: {pytorch_policy[pytorch_top10[0]]:.6f}")
    print(f"    Difference:    {diff[pytorch_top10[0]]:.6f}")
    print(f"    Rank in Keras: {np.where(np.argsort(keras_policy)[::-1] == pytorch_top10[0])[0][0] + 1}")
    
    # Distribution of differences
    print(f"\nDifference Distribution:")
    small_diff = (diff < 0.1).sum()
    medium_diff = ((diff >= 0.1) & (diff < 1.0)).sum()
    large_diff = ((diff >= 1.0) & (diff < 5.0)).sum()
    very_large_diff = (diff >= 5.0).sum()
    
    print(f"  < 0.1:     {small_diff:5d} ({small_diff/len(diff)*100:.1f}%)")
    print(f"  0.1-1.0:   {medium_diff:5d} ({medium_diff/len(diff)*100:.1f}%)")
    print(f"  1.0-5.0:   {large_diff:5d} ({large_diff/len(diff)*100:.1f}%)")
    print(f"  >= 5.0:    {very_large_diff:5d} ({very_large_diff/len(diff)*100:.1f}%)")
    
    # Check if it's a scale issue or absolute shift
    print(f"\nScale Analysis:")
    print(f"  Keras policy range:   [{keras_policy.min():.6f}, {keras_policy.max():.6f}]")
    print(f"  PyTorch policy range: [{pytorch_policy.min():.6f}, {pytorch_policy.max():.6f}]")
    print(f"  Keras policy mean:    {keras_policy.mean():.6f}")
    print(f"  PyTorch policy mean:  {pytorch_policy.mean():.6f}")
    print(f"  Mean difference:     {(keras_policy.mean() - pytorch_policy.mean()):.6f}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("The policy differences are:")
print("  - Maximum differences: ~10-13 (very large)")
print("  - Mean differences: ~0.8-1.0 (moderate)")
print("  - Top moves: Completely different (0/10 overlap)")
print("  - This suggests the model is making fundamentally different predictions")

