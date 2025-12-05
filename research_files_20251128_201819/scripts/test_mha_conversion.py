"""
Test MultiHeadAttention weight conversion to verify correctness.
"""

import zipfile
import tempfile
import h5py
import numpy as np
import os

keras_path = 'src/weights/best_model.keras'

with zipfile.ZipFile(keras_path, 'r') as zf:
    weights_file = next(f for f in zf.namelist() if f.endswith('.h5'))
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp.write(zf.read(weights_file))
        tmp_path = tmp.name

try:
    with h5py.File(tmp_path, 'r') as h5f:
        # Get Q, K, V weights
        q_kernel = np.array(h5f['layers/multi_head_attention/query_dense/vars/0'])
        k_kernel = np.array(h5f['layers/multi_head_attention/key_dense/vars/0'])
        v_kernel = np.array(h5f['layers/multi_head_attention/value_dense/vars/0'])
        q_bias = np.array(h5f['layers/multi_head_attention/query_dense/vars/1'])
        k_bias = np.array(h5f['layers/multi_head_attention/key_dense/vars/1'])
        v_bias = np.array(h5f['layers/multi_head_attention/value_dense/vars/1'])
        out_kernel = np.array(h5f['layers/multi_head_attention/output_dense/vars/0'])
        out_bias = np.array(h5f['layers/multi_head_attention/output_dense/vars/1'])
        
        print("="*70)
        print("MHA WEIGHT CONVERSION TEST")
        print("="*70)
        
        print(f"\nKeras weights:")
        print(f"  Q kernel: {q_kernel.shape}")
        print(f"  K kernel: {k_kernel.shape}")
        print(f"  V kernel: {v_kernel.shape}")
        print(f"  Q bias: {q_bias.shape}")
        print(f"  Out kernel: {out_kernel.shape}")
        
        # Current conversion approach
        print(f"\n{'='*70}")
        print("CURRENT CONVERSION:")
        print(f"{'='*70}")
        
        # Q, K, V: reshape (256, 8, 32) -> (256, 256), then transpose
        q_r = q_kernel.reshape(256, 256)
        k_r = k_kernel.reshape(256, 256)
        v_r = v_kernel.reshape(256, 256)
        
        # Transpose: (256, 256) -> (256, 256) for PyTorch format
        # PyTorch expects (out_dim, in_dim), Keras has (in_dim, out_dim)
        q_t = q_r.T  # (256, 256)
        k_t = k_r.T
        v_t = v_r.T
        
        # Concatenate: [Q; K; V] -> (768, 256)
        in_proj_weight = np.concatenate([q_t, k_t, v_t], axis=0)
        
        print(f"  Q reshaped: {q_r.shape} -> transposed: {q_t.shape}")
        print(f"  in_proj_weight shape: {in_proj_weight.shape}")
        print(f"    Rows 0-255: Q")
        print(f"    Rows 256-511: K")
        print(f"    Rows 512-767: V")
        
        # Bias: flatten (8, 32) -> (256,), then concatenate
        q_bias_flat = q_bias.flatten()  # (256,)
        k_bias_flat = k_bias.flatten()
        v_bias_flat = v_bias.flatten()
        in_proj_bias = np.concatenate([q_bias_flat, k_bias_flat, v_bias_flat])
        
        print(f"\n  Q bias flattened: {q_bias_flat.shape}")
        print(f"  in_proj_bias shape: {in_proj_bias.shape}")
        
        # Output projection: (8, 32, 256) -> (256, 256) -> transpose
        out_r = out_kernel.reshape(256, 256)
        out_t = out_r.T  # PyTorch format
        
        print(f"\n  Out kernel reshaped: {out_r.shape} -> transposed: {out_t.shape}")
        
        # Test with a dummy input
        print(f"\n{'='*70}")
        print("TESTING WITH DUMMY INPUT:")
        print(f"{'='*70}")
        
        # Simulate: input (1, 64, 256) -> through Q projection
        dummy_input = np.random.randn(1, 64, 256).astype(np.float32)
        
        # Keras-style computation (what we think it does):
        # For each head h: Q_h = input @ q_kernel[:, h, :]  # (1, 64, 256) @ (256, 32) = (1, 64, 32)
        # Then concatenate all heads: (1, 64, 8*32) = (1, 64, 256)
        
        # PyTorch-style computation:
        # Q = input @ in_proj_weight[0:256, :].T  # (1, 64, 256) @ (256, 256) = (1, 64, 256)
        
        # Let's test if they match
        print(f"  Input shape: {dummy_input.shape}")
        
        # Keras way: manual head computation
        q_keras_list = []
        for h in range(8):
            head_weights = q_kernel[:, h, :]  # (256, 32)
            head_output = dummy_input[0] @ head_weights  # (64, 32)
            q_keras_list.append(head_output)
        q_keras = np.concatenate(q_keras_list, axis=-1)  # (64, 256)
        
        # PyTorch way: single matrix
        q_pytorch = dummy_input[0] @ q_t.T  # (64, 256) @ (256, 256) = (64, 256)
        
        print(f"  Q Keras shape: {q_keras.shape}")
        print(f"  Q PyTorch shape: {q_pytorch.shape}")
        print(f"  Max difference: {np.abs(q_keras - q_pytorch).max():.6f}")
        print(f"  Mean difference: {np.abs(q_keras - q_pytorch).mean():.6f}")
        
        if np.allclose(q_keras, q_pytorch, atol=1e-5):
            print("  ✓ Q projection matches!")
        else:
            print("  ✗ Q projection DOES NOT match - conversion is wrong!")
        
        # Test output projection
        print(f"\n  Testing output projection:")
        # Keras: for each head, project (32,) -> (256,)
        # Then sum/average? Or concatenate?
        # Actually, Keras MHA typically concatenates heads, so input is (64, 256)
        dummy_mha_output = np.random.randn(64, 256).astype(np.float32)  # After attention, before out_proj
        
        # Keras way: manual head projection
        out_keras_list = []
        for h in range(8):
            head_weights = out_kernel[h, :, :]  # (32, 256)
            head_input = dummy_mha_output[:, h*32:(h+1)*32]  # (64, 32) - this head's output
            head_output = head_input @ head_weights  # (64, 256)
            out_keras_list.append(head_output)
        # Keras typically SUMS the head outputs (not concatenates for out_proj)
        out_keras = sum(out_keras_list)  # (64, 256)
        
        # PyTorch way: single matrix
        out_pytorch = dummy_mha_output @ out_t  # (64, 256) @ (256, 256) = (64, 256)
        
        print(f"  Out Keras shape: {out_keras.shape}")
        print(f"  Out PyTorch shape: {out_pytorch.shape}")
        print(f"  Max difference: {np.abs(out_keras - out_pytorch).max():.6f}")
        
        if np.allclose(out_keras, out_pytorch, atol=1e-4):
            print("  ✓ Output projection matches!")
        else:
            print("  ✗ Output projection DOES NOT match!")
            print("    Keras might SUM head outputs, PyTorch might do something else")
            print("    This could be the issue!")

finally:
    os.unlink(tmp_path)

