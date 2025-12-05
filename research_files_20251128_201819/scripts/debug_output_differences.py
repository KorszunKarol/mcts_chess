"""
Debug script to find where Keras and PyTorch outputs diverge.
Tests intermediate activations to pinpoint the issue.
"""

import sys
import os
import numpy as np
import chess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder

# Load pre-extracted Keras outputs
with open('saved_models/keras_outputs.json', 'r') as f:
    keras_outputs = json.load(f)

# Load PyTorch model
import torch
from src.transformer_model_pytorch import create_model

checkpoint = torch.load('saved_models/best_model_pytorch.pt', map_location='cpu')
pytorch_model = create_model()
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# Test on starting position
encoder = Encoder()
board = chess.Board()
position = encoder.encode(board)

# Convert to PyTorch format
input_tensor = np.transpose(position.copy(), (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = torch.from_numpy(input_tensor).float()

# Get Keras outputs
keras_data = keras_outputs['starting_position']
keras_value = np.array(keras_data["value"])
keras_policy = np.array(keras_data["policy"])

# Forward pass with hooks to capture intermediate values
intermediate_values = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            intermediate_values[name] = output.detach().cpu().numpy()
        elif isinstance(output, tuple):
            intermediate_values[name] = [o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
    return hook

# Register hooks at key points
hooks = []
hooks.append(pytorch_model.initial_conv.register_forward_hook(hook_fn('initial_conv')))
hooks.append(pytorch_model.initial_bn.register_forward_hook(hook_fn('initial_bn')))
hooks.append(pytorch_model.residual_blocks[0].register_forward_hook(hook_fn('residual_block_0')))
hooks.append(pytorch_model.residual_blocks[-1].register_forward_hook(hook_fn('residual_block_last')))
hooks.append(pytorch_model.transformer_layers[0].register_forward_hook(hook_fn('transformer_layer_0')))
hooks.append(pytorch_model.transformer_layers[-1].register_forward_hook(hook_fn('transformer_layer_last')))
hooks.append(pytorch_model.value_fc1.register_forward_hook(hook_fn('value_fc1')))
hooks.append(pytorch_model.policy_conv.register_forward_hook(hook_fn('policy_conv')))

# Run forward pass
with torch.no_grad():
    pytorch_value, pytorch_policy = pytorch_model(input_tensor)

pytorch_value = pytorch_value[0].numpy()
pytorch_policy = pytorch_policy[0].numpy()

# Remove hooks
for hook in hooks:
    hook.remove()

print("="*80)
print("OUTPUT DIFFERENCE ANALYSIS")
print("="*80)

print(f"\nValue Outputs:")
print(f"  Keras:   {keras_value}")
print(f"  PyTorch: {pytorch_value}")
print(f"  Max diff: {np.abs(keras_value - pytorch_value).max():.6f}")
print(f"  Mean diff: {np.abs(keras_value - pytorch_value).mean():.6f}")

print(f"\nPolicy Outputs:")
print(f"  Keras shape:   {keras_policy.shape}")
print(f"  PyTorch shape: {pytorch_policy.shape}")
print(f"  Max diff: {np.abs(keras_policy - pytorch_policy).max():.6f}")
print(f"  Mean diff: {np.abs(keras_policy - pytorch_policy).mean():.6f}")

print(f"\nPolicy Top-10:")
keras_top10 = np.argsort(keras_policy)[-10:][::-1]
pytorch_top10 = np.argsort(pytorch_policy)[-10:][::-1]
print(f"  Keras:   {keras_top10}")
print(f"  PyTorch: {pytorch_top10}")
print(f"  Overlap: {len(set(keras_top10) & set(pytorch_top10))}/10")

print(f"\n" + "="*80)
print("INTERMEDIATE ACTIVATION STATISTICS")
print("="*80)

for name, value in intermediate_values.items():
    if isinstance(value, np.ndarray):
        print(f"\n{name}:")
        print(f"  Shape: {value.shape}")
        print(f"  Mean: {value.mean():.6f}, Std: {value.std():.6f}")
        print(f"  Min: {value.min():.6f}, Max: {value.max():.6f}")
        if value.size < 100:
            print(f"  Values: {value.flatten()}")

# Check if policy head inputs are similar
print(f"\n" + "="*80)
print("POLICY HEAD ANALYSIS")
print("="*80)

policy_conv_out = intermediate_values.get('policy_conv')
if policy_conv_out is not None:
    print(f"\nPolicy Conv Output (before FC):")
    print(f"  Shape: {policy_conv_out.shape}")
    print(f"  Mean: {policy_conv_out.mean():.6f}, Std: {policy_conv_out.std():.6f}")
    print(f"  This should be (1, 2, 8, 8) = 128 values before flattening")
    
    # Check if the issue is in the conv or the FC
    policy_conv_flat = policy_conv_out.flatten()
    print(f"  Flattened shape: {policy_conv_flat.shape}")
    print(f"  First 10 values: {policy_conv_flat[:10]}")

print(f"\n" + "="*80)
print("HYPOTHESIS CHECK")
print("="*80)
print("If intermediate values look reasonable but final outputs differ,")
print("the issue is likely in:")
print("  1. MHA output projection conversion")
print("  2. Policy head weight conversion")
print("  3. Numerical precision accumulation")

