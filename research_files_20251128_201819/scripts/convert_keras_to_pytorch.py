"""
Weight Conversion Script: TensorFlow/Keras 3.x → PyTorch

This script transfers trained weights from a Keras 3.x model (.keras) to the 
equivalent PyTorch model.

Keras 3.x H5 weight naming convention:
    - layers/conv2d/vars/0 = kernel, vars/1 = bias
    - layers/batch_normalization/vars/0,1,2,3 = gamma, beta, moving_mean, moving_var
    - layers/multi_head_attention/query_dense/vars/0 = Q kernel
    - layers/layer_normalization/vars/0,1 = gamma, beta
"""

import argparse
import os
import sys
import zipfile
import tempfile
import logging
from typing import Dict, Optional
from collections import OrderedDict

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_weights_from_keras_file(keras_path: str) -> Dict[str, np.ndarray]:
    """Extract weights directly from a .keras file by reading the H5 weights file."""
    weights_dict = {}
    
    logger.info(f"Extracting weights from: {keras_path}")
    
    with zipfile.ZipFile(keras_path, 'r') as zf:
        file_list = zf.namelist()
        weights_file = next((f for f in file_list if f.endswith('.h5')), None)
        
        if not weights_file:
            raise ValueError(f"No .h5 weights file found in {keras_path}")
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(zf.read(weights_file))
            tmp_path = tmp.name
        
        try:
            with h5py.File(tmp_path, 'r') as h5f:
                def extract_datasets(group, prefix=''):
                    for key in group.keys():
                        item = group[key]
                        full_key = f"{prefix}/{key}" if prefix else key
                        
                        if isinstance(item, h5py.Dataset):
                            weights_dict[full_key] = np.array(item)
                        elif isinstance(item, h5py.Group):
                            extract_datasets(item, full_key)
                
                extract_datasets(h5f)
        finally:
            os.unlink(tmp_path)
    
    logger.info(f"✓ Extracted {len(weights_dict)} weight tensors")
    return weights_dict


def print_weight_structure(weights_dict: Dict[str, np.ndarray]):
    """Print the structure of extracted weights for debugging."""
    print("\n" + "="*70)
    print("EXTRACTED WEIGHT STRUCTURE")
    print("="*70)
    
    for key in sorted(weights_dict.keys()):
        if 'layers/' in key:
            print(f"  {key}: {weights_dict[key].shape}")


def transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """Keras (H,W,Cin,Cout) → PyTorch (Cout,Cin,H,W)"""
    return np.transpose(w, (3, 2, 0, 1))


def transpose_dense_weight(w: np.ndarray) -> np.ndarray:
    """Keras (In,Out) → PyTorch (Out,In)"""
    return np.transpose(w)


def get_weight(weights_dict: Dict, pattern: str) -> Optional[np.ndarray]:
    """Find a weight by pattern matching."""
    for key, value in weights_dict.items():
        if pattern in key:
            return value
    return None


def convert_weights(weights_dict: Dict[str, np.ndarray]) -> OrderedDict:
    """Convert Keras 3.x weights to PyTorch state dict format."""
    import torch
    
    state_dict = OrderedDict()
    
    logger.info("\n" + "="*60)
    logger.info("CONVERTING WEIGHTS")
    logger.info("="*60)
    
    # === 1. Initial Conv + BN ===
    logger.info("\n[1/5] Converting initial convolution...")
    
    w = get_weight(weights_dict, 'layers/conv2d/vars/0')
    if w is not None:
        state_dict['initial_conv.weight'] = torch.from_numpy(transpose_conv_weight(w).copy())
        logger.info("  ✓ initial_conv.weight")
    
    # Initial BatchNorm (vars: 0=gamma, 1=beta, 2=moving_mean, 3=moving_var)
    bn_prefix = 'layers/batch_normalization/vars'
    state_dict['initial_bn.weight'] = torch.from_numpy(get_weight(weights_dict, f'{bn_prefix}/0').copy())
    state_dict['initial_bn.bias'] = torch.from_numpy(get_weight(weights_dict, f'{bn_prefix}/1').copy())
    state_dict['initial_bn.running_mean'] = torch.from_numpy(get_weight(weights_dict, f'{bn_prefix}/2').copy())
    state_dict['initial_bn.running_var'] = torch.from_numpy(get_weight(weights_dict, f'{bn_prefix}/3').copy())
    logger.info("  ✓ initial_bn")
    
    # === 2. Residual Blocks ===
    logger.info("\n[2/5] Converting residual blocks...")
    
    # Block mapping: (block_idx, conv1_idx, conv2_idx, bn1_idx, bn2_idx, proj_idx)
    # The model has 4 residual blocks:
    # - Block 0,1: 128 filters (no projection)
    # - Block 2: 128→256 (has projection at conv2d_7)
    # - Block 3: 256 filters (no projection)
    
    block_map = [
        (0, '1', '2', '1', '2', None),      # conv2d_1,2 + bn_1,2
        (1, '3', '4', '3', '4', None),      # conv2d_3,4 + bn_3,4
        (2, '5', '6', '5', '6', '7'),       # conv2d_5,6 + bn_5,6 + proj(conv2d_7)
        (3, '8', '9', '7', '8', None),      # conv2d_8,9 + bn_7,8
    ]
    
    for block_idx, c1, c2, b1, b2, proj in block_map:
        prefix = f'residual_blocks.{block_idx}'
        
        # Conv1
        w = get_weight(weights_dict, f'layers/conv2d_{c1}/vars/0')
        if w is not None:
            state_dict[f'{prefix}.conv1.weight'] = torch.from_numpy(transpose_conv_weight(w).copy())
        
        # BN1
        bn_p = f'layers/batch_normalization_{b1}/vars'
        state_dict[f'{prefix}.bn1.weight'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/0').copy())
        state_dict[f'{prefix}.bn1.bias'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/1').copy())
        state_dict[f'{prefix}.bn1.running_mean'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/2').copy())
        state_dict[f'{prefix}.bn1.running_var'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/3').copy())
        
        # Conv2
        w = get_weight(weights_dict, f'layers/conv2d_{c2}/vars/0')
        if w is not None:
            state_dict[f'{prefix}.conv2.weight'] = torch.from_numpy(transpose_conv_weight(w).copy())
        
        # BN2
        bn_p = f'layers/batch_normalization_{b2}/vars'
        state_dict[f'{prefix}.bn2.weight'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/0').copy())
        state_dict[f'{prefix}.bn2.bias'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/1').copy())
        state_dict[f'{prefix}.bn2.running_mean'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/2').copy())
        state_dict[f'{prefix}.bn2.running_var'] = torch.from_numpy(get_weight(weights_dict, f'{bn_p}/3').copy())
        
        # Projection
        if proj:
            w = get_weight(weights_dict, f'layers/conv2d_{proj}/vars/0')
            if w is not None:
                state_dict[f'{prefix}.projection.weight'] = torch.from_numpy(transpose_conv_weight(w).copy())
        
        logger.info(f"  ✓ {prefix}")
    
    # === 3. Transformer Layers ===
    logger.info("\n[3/5] Converting transformer layers...")
    
    embed_dim = 256
    
    for layer_idx in range(6):
        pt_prefix = f'transformer_layers.{layer_idx}'
        
        # MHA naming: multi_head_attention, multi_head_attention_1, ...
        if layer_idx == 0:
            mha_prefix = 'layers/multi_head_attention'
        else:
            mha_prefix = f'layers/multi_head_attention_{layer_idx}'
        
        # Q, K, V kernels: (256, 8, 32) → reshape to (256, 256) → transpose
        q_kernel = get_weight(weights_dict, f'{mha_prefix}/query_dense/vars/0')
        k_kernel = get_weight(weights_dict, f'{mha_prefix}/key_dense/vars/0')
        v_kernel = get_weight(weights_dict, f'{mha_prefix}/value_dense/vars/0')
        
        if q_kernel is not None and k_kernel is not None and v_kernel is not None:
            q_r = q_kernel.reshape(embed_dim, embed_dim)
            k_r = k_kernel.reshape(embed_dim, embed_dim)
            v_r = v_kernel.reshape(embed_dim, embed_dim)
            
            in_proj_weight = np.concatenate([
                transpose_dense_weight(q_r),
                transpose_dense_weight(k_r),
                transpose_dense_weight(v_r)
            ], axis=0)
            state_dict[f'{pt_prefix}.attention.in_proj_weight'] = torch.from_numpy(in_proj_weight.copy())
        
        # Q, K, V biases: (8, 32) → flatten
        q_bias = get_weight(weights_dict, f'{mha_prefix}/query_dense/vars/1')
        k_bias = get_weight(weights_dict, f'{mha_prefix}/key_dense/vars/1')
        v_bias = get_weight(weights_dict, f'{mha_prefix}/value_dense/vars/1')
        
        if q_bias is not None and k_bias is not None and v_bias is not None:
            in_proj_bias = np.concatenate([
                q_bias.flatten(),
                k_bias.flatten(),
                v_bias.flatten()
            ])
            state_dict[f'{pt_prefix}.attention.in_proj_bias'] = torch.from_numpy(in_proj_bias.copy())
        
        # Output projection: (8, 32, 256) → (256, 256)
        # CORRECT CONVERSION (per research): Reshape then Transpose
        # Keras: (8, 32, 256) = (num_heads, key_dim, embed_dim)
        # Step 1: Reshape to collapse head and key dimensions: (8*32, 256) = (256, 256)
        # Step 2: Transpose for PyTorch format: (256, 256) -> (256, 256) [out_features, in_features]
        out_kernel = get_weight(weights_dict, f'{mha_prefix}/output_dense/vars/0')
        out_bias = get_weight(weights_dict, f'{mha_prefix}/output_dense/vars/1')
        
        if out_kernel is not None:
            # EXACT conversion per document: reshape then transpose
            # Step 1: Reshape to collapse head and key dimensions
            # (8, 32, 256) -> (256, 256) using standard NumPy reshape (C-order)
            out_reshaped = np.reshape(out_kernel, (-1, out_kernel.shape[-1]))
            # Step 2: Transpose for PyTorch Linear layer format (out_features, in_features)
            out_proj_weight = np.transpose(out_reshaped)
            # Ensure float32 precision
            state_dict[f'{pt_prefix}.attention.out_proj.weight'] = torch.from_numpy(
                out_proj_weight.astype(np.float32).copy()
            )
        if out_bias is not None:
            # Ensure float32 precision
            state_dict[f'{pt_prefix}.attention.out_proj.bias'] = torch.from_numpy(
                out_bias.astype(np.float32).copy()
            )
        
        # LayerNorm 1: layer_normalization, layer_normalization_2, layer_normalization_4, ...
        # Pattern: ln_idx = layer_idx * 2
        ln1_idx = layer_idx * 2
        if ln1_idx == 0:
            ln1_prefix = 'layers/layer_normalization/vars'
        else:
            ln1_prefix = f'layers/layer_normalization_{ln1_idx}/vars'
        
        ln1_gamma = get_weight(weights_dict, f'{ln1_prefix}/0')
        ln1_beta = get_weight(weights_dict, f'{ln1_prefix}/1')
        if ln1_gamma is not None:
            state_dict[f'{pt_prefix}.norm1.weight'] = torch.from_numpy(ln1_gamma.copy())
        if ln1_beta is not None:
            state_dict[f'{pt_prefix}.norm1.bias'] = torch.from_numpy(ln1_beta.copy())
        
        # FFN: dense_{layer_idx*2}, dense_{layer_idx*2+1}
        d1_idx = layer_idx * 2
        d2_idx = layer_idx * 2 + 1
        
        if d1_idx == 0:
            d1_prefix = 'layers/dense/vars'
        else:
            d1_prefix = f'layers/dense_{d1_idx}/vars'
        d2_prefix = f'layers/dense_{d2_idx}/vars'
        
        d1_kernel = get_weight(weights_dict, f'{d1_prefix}/0')
        d1_bias = get_weight(weights_dict, f'{d1_prefix}/1')
        d2_kernel = get_weight(weights_dict, f'{d2_prefix}/0')
        d2_bias = get_weight(weights_dict, f'{d2_prefix}/1')
        
        if d1_kernel is not None:
            state_dict[f'{pt_prefix}.ffn.0.weight'] = torch.from_numpy(transpose_dense_weight(d1_kernel).copy())
        if d1_bias is not None:
            state_dict[f'{pt_prefix}.ffn.0.bias'] = torch.from_numpy(d1_bias.copy())
        if d2_kernel is not None:
            state_dict[f'{pt_prefix}.ffn.3.weight'] = torch.from_numpy(transpose_dense_weight(d2_kernel).copy())
        if d2_bias is not None:
            state_dict[f'{pt_prefix}.ffn.3.bias'] = torch.from_numpy(d2_bias.copy())
        
        # LayerNorm 2: layer_normalization_1, layer_normalization_3, ...
        ln2_idx = layer_idx * 2 + 1
        ln2_prefix = f'layers/layer_normalization_{ln2_idx}/vars'
        
        ln2_gamma = get_weight(weights_dict, f'{ln2_prefix}/0')
        ln2_beta = get_weight(weights_dict, f'{ln2_prefix}/1')
        if ln2_gamma is not None:
            state_dict[f'{pt_prefix}.norm2.weight'] = torch.from_numpy(ln2_gamma.copy())
        if ln2_beta is not None:
            state_dict[f'{pt_prefix}.norm2.bias'] = torch.from_numpy(ln2_beta.copy())
        
        logger.info(f"  ✓ {pt_prefix}")
    
    # === 4. Value Head ===
    logger.info("\n[4/5] Converting value head...")
    
    # dense_12 → value_fc1 (256 → 256)
    v_fc1_kernel = get_weight(weights_dict, 'layers/dense_12/vars/0')
    v_fc1_bias = get_weight(weights_dict, 'layers/dense_12/vars/1')
    if v_fc1_kernel is not None:
        state_dict['value_fc1.weight'] = torch.from_numpy(transpose_dense_weight(v_fc1_kernel).copy())
        logger.info("  ✓ value_fc1.weight")
    if v_fc1_bias is not None:
        state_dict['value_fc1.bias'] = torch.from_numpy(v_fc1_bias.copy())
        logger.info("  ✓ value_fc1.bias")
    
    # dense_13 → value_fc2 (256 → 3)
    v_fc2_kernel = get_weight(weights_dict, 'layers/dense_13/vars/0')
    v_fc2_bias = get_weight(weights_dict, 'layers/dense_13/vars/1')
    if v_fc2_kernel is not None:
        state_dict['value_fc2.weight'] = torch.from_numpy(transpose_dense_weight(v_fc2_kernel).copy())
        logger.info("  ✓ value_fc2.weight")
    if v_fc2_bias is not None:
        state_dict['value_fc2.bias'] = torch.from_numpy(v_fc2_bias.copy())
        logger.info("  ✓ value_fc2.bias")
    
    # === 5. Policy Head ===
    logger.info("\n[5/5] Converting policy head...")
    
    # conv2d_10 → policy_conv (256 → 2)
    p_conv_kernel = get_weight(weights_dict, 'layers/conv2d_10/vars/0')
    p_conv_bias = get_weight(weights_dict, 'layers/conv2d_10/vars/1')
    if p_conv_kernel is not None:
        state_dict['policy_conv.weight'] = torch.from_numpy(transpose_conv_weight(p_conv_kernel).copy())
        logger.info("  ✓ policy_conv.weight")
    if p_conv_bias is not None:
        state_dict['policy_conv.bias'] = torch.from_numpy(p_conv_bias.copy())
        logger.info("  ✓ policy_conv.bias")
    
    # dense_14 → policy_fc (128 → 4672)
    p_fc_kernel = get_weight(weights_dict, 'layers/dense_14/vars/0')
    p_fc_bias = get_weight(weights_dict, 'layers/dense_14/vars/1')
    if p_fc_kernel is not None:
        state_dict['policy_fc.weight'] = torch.from_numpy(transpose_dense_weight(p_fc_kernel).copy())
        logger.info("  ✓ policy_fc.weight")
    if p_fc_bias is not None:
        state_dict['policy_fc.bias'] = torch.from_numpy(p_fc_bias.copy())
        logger.info("  ✓ policy_fc.bias")
    
    return state_dict


def load_pytorch_model():
    """Create a fresh PyTorch model to receive weights."""
    from src.transformer_model_pytorch import create_model
    model = create_model()
    logger.info("✓ PyTorch model created")
    return model


def convert_model(keras_path: str, output_path: str, debug: bool = False):
    """Full conversion from Keras to PyTorch."""
    print("="*60)
    print("KERAS → PYTORCH WEIGHT CONVERSION")
    print("="*60)
    
    weights_dict = extract_weights_from_keras_file(keras_path)
    
    if debug:
        print_weight_structure(weights_dict)
        return
    
    import torch
    
    pytorch_model = load_pytorch_model()
    converted_state_dict = convert_weights(weights_dict)
    
    logger.info("\n" + "="*60)
    logger.info("LOADING WEIGHTS INTO PYTORCH MODEL")
    logger.info("="*60)
    
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(
        converted_state_dict, strict=False
    )
    
    if missing_keys:
        logger.warning(f"\nMissing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:
            logger.warning(f"  - {key}")
        if len(missing_keys) > 10:
            logger.warning(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        logger.warning(f"\nUnexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:
            logger.warning(f"  - {key}")
    
    total_params = len(list(pytorch_model.state_dict().keys()))
    loaded_params = total_params - len(missing_keys)
    coverage = loaded_params / total_params * 100
    
    logger.info(f"\n✓ Loaded {loaded_params}/{total_params} parameters ({coverage:.1f}%)")
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'source_keras_model': keras_path,
    }, output_path)
    
    logger.info(f"✓ PyTorch model saved to: {output_path}")
    
    # Sanity check
    logger.info("\n" + "="*60)
    logger.info("SANITY CHECK")
    logger.info("="*60)
    
    pytorch_model.eval()
    dummy_input = torch.randn(1, 34, 8, 8)
    
    with torch.no_grad():
        value, policy = pytorch_model(dummy_input)
    
    logger.info(f"  Input shape: {dummy_input.shape}")
    logger.info(f"  Value output: {value.shape}, sum={value.sum().item():.4f}")
    logger.info(f"  Policy output: {policy.shape}")
    logger.info("  ✓ Model runs successfully!")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    
    return pytorch_model


def main():
    parser = argparse.ArgumentParser(description="Convert Keras → PyTorch")
    parser.add_argument('--keras-model', type=str, default='src/weights/best_model.keras')
    parser.add_argument('--output', type=str, default='saved_models/best_model_pytorch.pt')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    convert_model(args.keras_model, args.output, args.debug)


if __name__ == '__main__':
    main()
