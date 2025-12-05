"""
JAX <-> PyTorch bridge utilities.

This module provides zero-copy tensor transfer between JAX and PyTorch
using DLPack, enabling efficient interoperability between frameworks.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging

import jax
import jax.numpy as jnp
from src.utils import sentinel

logger = logging.getLogger(__name__)


def torch_to_jax(tensor) -> jnp.ndarray:
    """
    Convert PyTorch tensor to JAX array using DLPack (zero-copy).
    
    Args:
        tensor: PyTorch tensor (must be on CUDA).
        
    Returns:
        JAX array sharing the same memory.
        
    Note:
        The tensor must be contiguous and on CUDA for zero-copy.
        CPU tensors will be copied.
    """
    import torch
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if sentinel.enabled:
        sentinel.log_tensor("torch_to_jax", tensor)
    
    # Use DLPack for zero-copy transfer
    if tensor.is_cuda:
        dlpack = torch.utils.dlpack.to_dlpack(tensor)
        out = jax.dlpack.from_dlpack(dlpack)
    else:
        # CPU path: copy through numpy
        out = jnp.array(tensor.detach().numpy())

    if sentinel.enabled:
        sentinel.log_tensor("torch_to_jax_out", out)
    return out


def jax_to_torch(array: jnp.ndarray, device: str = "cuda"):
    """
    Convert JAX array to PyTorch tensor using DLPack (zero-copy).
    
    Args:
        array: JAX array.
        device: Target PyTorch device.
        
    Returns:
        PyTorch tensor sharing the same memory (if on GPU).
    """
    import torch
    
    # Ensure array is on device
    array = jax.device_put(array)

    if sentinel.enabled:
        sentinel.log_tensor("jax_to_torch_in", array)
    
    # Use DLPack for zero-copy transfer
    dlpack = jax.dlpack.to_dlpack(array)
    tensor = torch.utils.dlpack.from_dlpack(dlpack)
    tensor = tensor.to(device)

    if sentinel.enabled:
        sentinel.log_tensor("jax_to_torch_out", tensor)
    return tensor


def pytorch_state_dict_to_flax(
    state_dict: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert PyTorch state dict to Flax parameter structure.
    
    Args:
        state_dict: PyTorch model state dict.
        model_config: Optional model configuration.
        
    Returns:
        Flax-compatible parameter dictionary.
    """
    import torch
    
    flax_params = {}
    
    for key, value in state_dict.items():
        # Convert tensor to numpy
        if isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = value
        
        # Parse the key to build nested structure
        # PyTorch: "initial_conv.weight" -> Flax: {"initial_conv": {"kernel": ...}}
        parts = key.split(".")
        
        # Navigate/create nested dict
        current = flax_params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Handle naming conventions
        final_key = parts[-1]
        
        # PyTorch "weight" -> Flax "kernel" for linear/conv
        if final_key == "weight":
            # Check if this is a conv or linear layer
            if np_value.ndim == 4:
                # Conv2d: PyTorch (out, in, h, w) -> Flax (h, w, in, out)
                np_value = np_value.transpose(2, 3, 1, 0)
                final_key = "kernel"
            elif np_value.ndim == 2:
                # Linear: PyTorch (out, in) -> Flax (in, out)
                np_value = np_value.T
                final_key = "kernel"
        
        # BatchNorm naming
        if final_key == "running_mean":
            final_key = "mean"
        elif final_key == "running_var":
            final_key = "var"
        
        current[final_key] = jnp.array(np_value)
    
    return flax_params


def load_pytorch_weights_to_flax(
    pytorch_path: str,
    flax_model,
) -> Dict[str, Any]:
    """
    Load PyTorch weights file and convert to Flax parameters.
    
    Args:
        pytorch_path: Path to PyTorch .pt file.
        flax_model: Flax model instance (for shape validation).
        
    Returns:
        Flax parameter dictionary.
    """
    import torch
    
    logger.info(f"Loading PyTorch weights from {pytorch_path}")
    
    # Load PyTorch state dict
    state_dict = torch.load(pytorch_path, map_location="cpu")
    
    # Handle different save formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Convert to Flax format
    flax_params = pytorch_state_dict_to_flax(state_dict)
    
    logger.info(f"Converted {len(state_dict)} PyTorch parameters to Flax")
    
    return flax_params

