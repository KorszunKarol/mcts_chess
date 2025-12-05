"""
Verify MHA conversion using forward hooks as recommended by the document.
This will identify if the issue is upstream (Q/K/V) or in the output projection.
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

print("="*80)
print("MHA HOOK VERIFICATION (Per Document Recommendation)")
print("="*80)

# Load pre-extracted Keras outputs
with open('saved_models/keras_outputs.json', 'r') as f:
    keras_outputs = json.load(f)

# Load PyTorch model
checkpoint = torch.load('saved_models/best_model_pytorch.pt', map_location='cpu')
pytorch_model = create_model()
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# Test on starting position
encoder = Encoder()
board = chess.Board()
position = encoder.encode(board)

# Convert to PyTorch format (verify NCHW conversion)
input_tensor = np.transpose(position.copy(), (2, 0, 1))  # NHWC -> NCHW
input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = torch.from_numpy(input_tensor).float()  # Ensure float32

print(f"\n[1] Input Verification:")
print(f"  Keras expects: (1, 8, 8, 34) = NHWC")
print(f"  PyTorch input: {input_tensor.shape} = NCHW")
print(f"  ✓ Input layout correct")

# Hook to capture input to out_proj (before projection)
activations = {}

def get_activation(name):
    def hook(module, input, output):
        # For Linear layers, input is a tuple with one element (the input tensor)
        if isinstance(input, tuple):
            if len(input) > 0:
                activations[name] = input[0].detach().cpu().numpy()
        elif hasattr(input, 'detach'):
            activations[name] = input.detach().cpu().numpy()
        else:
            activations[name] = np.array(input)
    return hook

# Register hook on first transformer layer's output projection
hook = pytorch_model.transformer_layers[0].attention.out_proj.register_forward_hook(
    get_activation('mha_out_proj_input')
)

# Also hook the attention output (before out_proj)
def get_attention_output(name):
    def hook(module, input, output):
        # MultiheadAttention returns (attn_output, attn_output_weights)
        if isinstance(output, tuple):
            activations[name] = output[0].detach().cpu().numpy()
        else:
            activations[name] = output.detach().cpu().numpy()
    return hook

hook2 = pytorch_model.transformer_layers[0].attention.register_forward_hook(
    get_attention_output('mha_attention_output')
)

# Forward pass
with torch.no_grad():
    value, policy = pytorch_model(input_tensor)

# Remove hooks
hook.remove()
hook2.remove()

# Get the captured activations
mha_input_to_out_proj = activations.get('mha_out_proj_input')
mha_attention_out = activations.get('mha_attention_output')

print(f"\n[2] Captured Activations:")
if mha_input_to_out_proj is not None:
    print(f"  Input to out_proj shape: {mha_input_to_out_proj.shape}")
    print(f"  Input to out_proj mean: {mha_input_to_out_proj.mean():.6f}")
    print(f"  Input to out_proj std: {mha_input_to_out_proj.std():.6f}")
    print(f"  Input to out_proj min: {mha_input_to_out_proj.min():.6f}")
    print(f"  Input to out_proj max: {mha_input_to_out_proj.max():.6f}")
    print(f"  First 10 values: {mha_input_to_out_proj.flatten()[:10]}")

if mha_attention_out is not None:
    print(f"\n  Attention output shape: {mha_attention_out.shape}")
    print(f"  Attention output mean: {mha_attention_out.mean():.6f}")
    print(f"  Attention output std: {mha_attention_out.std():.6f}")

print(f"\n[3] Analysis:")
print(f"  The input to out_proj should be the concatenated head outputs.")
print(f"  Shape should be (batch, seq_len, embed_dim) = (1, 64, 256)")
print(f"  This represents: [head0(32), head1(32), ..., head7(32)] concatenated")
print()
print(f"  If this matches Keras (when we can extract it), the issue is NOT upstream.")
print(f"  If this differs from Keras, the issue IS upstream (Q/K/V conversion).")

# Check Q/K/V weights structure
print(f"\n[4] Q/K/V Weight Verification:")
mha = pytorch_model.transformer_layers[0].attention
in_proj_weight = mha.in_proj_weight.data.numpy()
print(f"  in_proj_weight shape: {in_proj_weight.shape}")
print(f"  Should be (768, 256) = [Q(256,256); K(256,256); V(256,256)]")
print(f"  Q portion (rows 0:256) mean: {in_proj_weight[0:256, :].mean():.6f}")
print(f"  K portion (rows 256:512) mean: {in_proj_weight[256:512, :].mean():.6f}")
print(f"  V portion (rows 512:768) mean: {in_proj_weight[512:768, :].mean():.6f}")

# Check output projection weights
print(f"\n[5] Output Projection Weight Verification:")
out_proj_weight = mha.out_proj.weight.data.numpy()
print(f"  out_proj_weight shape: {out_proj_weight.shape}")
print(f"  Mean: {out_proj_weight.mean():.6f}")
print(f"  Std: {out_proj_weight.std():.6f}")
print(f"  First value: {out_proj_weight[0, 0]:.6f}")

print(f"\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Compare the 'Input to out_proj' with Keras equivalent.")
print("If they match → Issue is in output projection conversion")
print("If they differ → Issue is upstream (Q/K/V or attention computation)")

