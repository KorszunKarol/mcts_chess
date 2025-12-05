"""
JAX <-> PyTorch bridge utilities.

Provides zero-copy tensor transfer and robust weight conversion.
"""

from __future__ import annotations

from typing import Dict, Any
import logging
import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def torch_to_jax(tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array using DLPack."""
    import torch
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tensor.is_cuda:
        dlpack = torch.utils.dlpack.to_dlpack(tensor)
        out = jax.dlpack.from_dlpack(dlpack)
    else:
        out = jnp.array(tensor.detach().numpy())
    return out


def jax_to_torch(array: jnp.ndarray, device: str = "cuda"):
    """Convert JAX array to PyTorch tensor using DLPack."""
    import torch
    array = jax.device_put(array)
    dlpack = jax.dlpack.to_dlpack(array)
    tensor = torch.utils.dlpack.from_dlpack(dlpack)
    return tensor.to(device)


def pytorch_state_dict_to_flax(state_dict: Dict[str, Any], num_heads: int | None = None) -> Dict[str, Any]:
    """
    Convert PyTorch state dict to Flax variable structure (params + batch_stats).
    """
    params: Dict[str, Any] = {}
    batch_stats: Dict[str, Any] = {}

    def convert_weight(key: str, tensor):
        np_val = tensor.detach().cpu().numpy()
        if np_val.ndim == 4:
            # Conv2d: (Out, In, H, W) -> (H, W, In, Out)
            return np_val.transpose(2, 3, 1, 0)
        if np_val.ndim == 2:
            # Linear: (Out, In) -> (In, Out)
            return np_val.T
        return np_val

    def _assign(target_dict: Dict[str, Any], path: str, value) -> None:
        parts = path.split(".")
        curr = target_dict
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = jnp.array(value)

    for key, value in state_dict.items():
        # 1) Normalize block naming
        key = key.replace("residual_blocks.", "residual_block_")
        key = key.replace("transformer_layers.", "transformer_block_")

        # 2) Multihead Attention handling
        if "in_proj_weight" in key:
            w_q, w_k, w_v = np.array_split(value.detach().cpu().numpy(), 3, axis=0)
            embed_dim = w_q.shape[1]
            heads = num_heads or 8
            head_dim = embed_dim // heads
            base = key.replace("attention.in_proj_weight", "attention")
            _assign(params, f"{base}.query.kernel", w_q.T.reshape(embed_dim, heads, head_dim))
            _assign(params, f"{base}.key.kernel", w_k.T.reshape(embed_dim, heads, head_dim))
            _assign(params, f"{base}.value.kernel", w_v.T.reshape(embed_dim, heads, head_dim))
            continue

        if "in_proj_bias" in key:
            b_q, b_k, b_v = np.array_split(value.detach().cpu().numpy(), 3, axis=0)
            embed_dim = b_q.shape[0]
            heads = num_heads or 8
            head_dim = embed_dim // heads
            base = key.replace("attention.in_proj_bias", "attention")
            _assign(params, f"{base}.query.bias", b_q.reshape(heads, head_dim))
            _assign(params, f"{base}.key.bias", b_k.reshape(heads, head_dim))
            _assign(params, f"{base}.value.bias", b_v.reshape(heads, head_dim))
            continue

        if "out_proj.weight" in key:
            heads = num_heads or 8
            embed_dim = value.shape[1]
            head_dim = embed_dim // heads
            base = key.replace("out_proj.weight", "out.kernel")
            _assign(params, base, value.detach().cpu().numpy().T.reshape(heads, head_dim, embed_dim))
            continue

        if "out_proj.bias" in key:
            base = key.replace("out_proj.bias", "out.bias")
            _assign(params, base, value.detach().cpu().numpy())
            continue

        # 3) FFN mapping
        if "ffn.0.weight" in key:
            base = key.replace("ffn.0.weight", "ffn_1.kernel")
            _assign(params, base, value.detach().cpu().numpy().T)
            continue

        if "ffn.0.bias" in key:
            base = key.replace("ffn.0.bias", "ffn_1.bias")
            _assign(params, base, value.detach().cpu().numpy())
            continue

        if "ffn.3.weight" in key:
            base = key.replace("ffn.3.weight", "ffn_2.kernel")
            _assign(params, base, value.detach().cpu().numpy().T)
            continue

        if "ffn.3.bias" in key:
            base = key.replace("ffn.3.bias", "ffn_2.bias")
            _assign(params, base, value.detach().cpu().numpy())
            continue

        if "linear1" in key:
            key = key.replace("linear1", "ffn_1")
        if "linear2" in key:
            key = key.replace("linear2", "ffn_2")

        # 4) Sort into params vs batch_stats
        key_parts = key.split(".")
        last = key_parts[-1]
        target_dict = params
        final_val = value

        if last == "weight":
            if "norm" in key_parts[-2] or "bn" in key_parts[-2]:
                key_parts[-1] = "scale"
                final_val = value.detach().cpu().numpy()
            else:
                key_parts[-1] = "kernel"
                final_val = convert_weight(key, value)
        elif last == "bias":
            final_val = value.detach().cpu().numpy()
        elif last == "running_mean":
            key_parts[-1] = "mean"
            target_dict = batch_stats
            final_val = value.detach().cpu().numpy()
        elif last == "running_var":
            key_parts[-1] = "var"
            target_dict = batch_stats
            final_val = value.detach().cpu().numpy()
        elif "num_batches_tracked" in last:
            continue

        final_key = ".".join(key_parts)
        _assign(target_dict, final_key, final_val)

    return {"params": params, "batch_stats": batch_stats}


def load_pytorch_weights_to_flax(pytorch_path: str, flax_model) -> Dict[str, Any]:
    """Load PyTorch weights and return Flax variables dict."""
    import torch

    logger.info(f"Loading PyTorch weights from {pytorch_path}")
    state_dict = torch.load(pytorch_path, map_location="cpu")

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    num_heads = getattr(flax_model, "num_heads", None)
    variables = pytorch_state_dict_to_flax(state_dict, num_heads=num_heads)
    logger.info("Converted PyTorch parameters to Flax variables")
    return variables

