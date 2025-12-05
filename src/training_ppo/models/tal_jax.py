"""
Flax/JAX implementation of the Tal Model (HybridChessModel).

This is a port of the PyTorch transformer_model_pytorch.py to Flax,
enabling use with DeepMind's mctx library for batched MCTS.

Architecture:
    - CNN Stem: 4 residual blocks (34 -> 128 -> 128 -> 256 -> 256)
    - Transformer Body: 6 encoder layers (256 dim, 8 heads)
    - Value Head: Global avg pool -> FC256 -> FC3 (W/D/L)
    - Policy Head: Reshape -> Conv1x1 -> Flatten -> FC4672
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, Dict, Any, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from src.utils import sentinel

logger = logging.getLogger(__name__)


# Action space size (same as PyTorch model)
ACTION_SPACE_SIZE = 4672


class ModelOutput(NamedTuple):
    """Output from TalModelJAX forward pass."""
    value: jnp.ndarray        # (B, 3) W/D/L probabilities
    policy_logits: jnp.ndarray  # (B, 4672) raw action logits
    
    def get_scalar_value(self) -> jnp.ndarray:
        """Convert W/D/L probs to scalar value in [-1, 1]."""
        # value[:, 2] = Win prob, value[:, 0] = Loss prob
        return self.value[:, 2] - self.value[:, 0]


class ResidualBlock(nn.Module):
    """
    Residual block for CNN stem.
    
    Structure: Conv -> BN -> SiLU -> Conv -> BN -> (+ residual) -> SiLU
    """
    out_channels: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]  # Flax uses NHWC
        
        residual = x
        
        # First conv
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.silu(x)
        
        # Second conv
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
        
        # Projection if needed
        if in_channels != self.out_channels:
            residual = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                use_bias=False,
                name="projection",
            )(residual)
        
        x = x + residual
        x = nn.silu(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with post-LN architecture.
    
    Structure: MultiHeadAttention -> Dropout -> (+) -> LN -> FFN -> (+) -> LN
    """
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            name="attention",
        )(x, x)
        
        attn_out = nn.Dropout(rate=self.dropout, deterministic=not train)(attn_out)
        x = nn.LayerNorm(epsilon=1e-6, name="norm1")(x + attn_out)
        
        # FFN
        ffn_out = nn.Dense(self.ff_dim, name="ffn_dense1")(x)
        ffn_out = nn.silu(ffn_out)
        ffn_out = nn.Dropout(rate=self.dropout, deterministic=not train)(ffn_out)
        ffn_out = nn.Dense(self.embed_dim, name="ffn_dense2")(ffn_out)
        
        x = nn.LayerNorm(epsilon=1e-6, name="norm2")(x + ffn_out)
        
        return x


class TalModelJAX(nn.Module):
    """
    Flax implementation of the Tal Model (Hybrid CNN-Transformer).
    
    This matches the architecture of transformer_model_pytorch.py for
    weight compatibility.
    
    Input: (B, 8, 8, 34) - Flax NHWC format
    Outputs:
        value: (B, 3) - W/D/L probabilities
        policy_logits: (B, 4672) - raw action logits
    """
    input_channels: int = 34
    action_space_size: int = ACTION_SPACE_SIZE
    stem_filters: Tuple[int, ...] = (128, 128, 256, 256)
    num_transformer_layers: int = 6
    num_heads: int = 8
    key_dim: int = 32
    ff_dim: int = 1024
    dropout: float = 0.1
    
    @sentinel.trace
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        train: bool = True,
    ) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            x: (B, 8, 8, 34) input in NHWC format, or (B, 34, 8, 8) in NCHW.
            train: Whether in training mode (affects dropout/batchnorm).
            
        Returns:
            ModelOutput with value probabilities and policy logits.
        """
        batch_size = x.shape[0]
        embed_dim = self.num_heads * self.key_dim  # 256
        
        if sentinel.enabled:
            sentinel.log_tensor("input", x)

        # Handle NCHW input (convert to NHWC for Flax)
        if x.shape[1] == self.input_channels:  # NCHW format
            x = jnp.transpose(x, (0, 2, 3, 1))  # -> NHWC
        
        # === CNN Stem ===
        # Initial convolution
        x = nn.Conv(
            features=self.stem_filters[0],
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            name="initial_conv",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="initial_bn")(x)
        x = nn.silu(x)
        
        # Residual blocks
        for i, filters in enumerate(self.stem_filters):
            x = ResidualBlock(out_channels=filters, name=f"residual_block_{i}")(x, train=train)
        if sentinel.enabled:
            sentinel.log_tensor("stem_output", x)
        
        # x shape: (B, 8, 8, 256)
        
        # === Prepare for Transformer ===
        # Flatten spatial: (B, 8, 8, 256) -> (B, 64, 256)
        x = x.reshape(batch_size, 64, embed_dim)
        if sentinel.enabled:
            sentinel.log_tensor("transformer_input", x)
        
        # === Transformer Body ===
        for i in range(self.num_transformer_layers):
            x = TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                name=f"transformer_block_{i}",
            )(x, train=train)
        if sentinel.enabled:
            sentinel.log_tensor("transformer_output", x)
        
        # x shape: (B, 64, 256)
        
        # === Value Head ===
        # Global average pooling
        value_repr = jnp.mean(x, axis=1)  # (B, 256)
        value_out = nn.Dense(256, name="value_fc1")(value_repr)
        value_out = nn.silu(value_out)
        value_out = nn.Dense(3, name="value_fc2")(value_out)
        value_probs = nn.softmax(value_out, axis=-1)  # (B, 3)
        
        # === Policy Head ===
        # CRITICAL: Must flatten before final Dense layer to avoid shape mismatch
        # Transformer output x has shape (B, 64, 256) - DO NOT use directly in Dense layer
        
        # 1. Reshape sequence back to spatial grid: (B, 64, 256) -> (B, 8, 8, 256)
        policy_spatial = x.reshape(batch_size, 8, 8, embed_dim)
        
        # 2. Conv 1x1 (Project to 2 channels)
        policy_out = nn.Conv(
            features=2,
            kernel_size=(1, 1),
            name="policy_conv",
        )(policy_spatial)
        policy_out = nn.silu(policy_out)  # Shape: (B, 8, 8, 2)
        
        # 3. Flatten spatial dims: (B, 8, 8, 2) -> (B, 128)
        # This is CRITICAL - Dense layer requires 2D input (B, features)
        policy_out = policy_out.reshape(batch_size, -1)  # Shape: (B, 128)
        
        # 4. Final Dense Layer - MUST use policy_out (flattened), NOT x (sequence)
        # Using x here would cause shape (B, 64, 4672) error
        policy_logits = nn.Dense(
            self.action_space_size,
            name="policy_fc",
        )(policy_out)  # Correct Shape: (B, 4672)

        if sentinel.enabled:
            sentinel.log_tensor("value_probs", value_probs)
            sentinel.log_tensor("policy_logits", policy_logits)
        
        return ModelOutput(value=value_probs, policy_logits=policy_logits)
    
    @classmethod
    def from_pytorch(
        cls,
        weights_path: str,
        **kwargs,
    ) -> Tuple["TalModelJAX", Dict[str, Any]]:
        """
        Create model and load weights from PyTorch checkpoint.
        
        Args:
            weights_path: Path to PyTorch .pt file.
            **kwargs: Additional model config.
            
        Returns:
            Tuple of (model instance, loaded parameters).
        """
        from src.training_ppo.models.jax_bridge import load_pytorch_weights_to_flax
        
        # Create model
        model = cls(**kwargs)
        
        # Initialize with dummy input to get param structure
        # Note: init() doesn't take train parameter - it's only for apply()
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 8, 8, 34))
        variables = model.init(key, dummy_input)
        
        # Load and convert PyTorch weights
        pytorch_params = load_pytorch_weights_to_flax(weights_path, model)
        
        # Merge with initialized params (for any missing keys)
        params = _merge_params(variables["params"], pytorch_params)
        
        logger.info(f"Loaded TalModelJAX from {weights_path}")
        
        return model, {"params": params, "batch_stats": variables.get("batch_stats", {})}


def _merge_params(
    base_params: Dict[str, Any],
    new_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge new parameters into base, keeping base values for missing keys.
    
    Args:
        base_params: Base parameter structure (from init).
        new_params: New parameter values (from PyTorch conversion).
        
    Returns:
        Merged parameter dictionary.
    """
    result = {}
    
    for key in base_params:
        if key in new_params:
            if isinstance(base_params[key], dict):
                result[key] = _merge_params(base_params[key], new_params[key])
            else:
                result[key] = new_params[key]
        else:
            result[key] = base_params[key]
            logger.warning(f"Parameter {key} not found in PyTorch weights, using initialized value")
    
    return result


def create_model(
    config: Optional[Dict[str, Any]] = None,
) -> TalModelJAX:
    """
    Factory function to create TalModelJAX.
    
    Args:
        config: Optional model configuration.
        
    Returns:
        TalModelJAX instance.
    """
    if config is None:
        config = {}
    
    return TalModelJAX(
        input_channels=config.get("input_channels", 34),
        action_space_size=config.get("action_space_size", ACTION_SPACE_SIZE),
        stem_filters=config.get("stem_filters", (128, 128, 256, 256)),
        num_transformer_layers=config.get("num_transformer_layers", 6),
        num_heads=config.get("num_heads", 8),
        key_dim=config.get("key_dim", 32),
        ff_dim=config.get("ff_dim", 1024),
        dropout=config.get("dropout", 0.1),
    )


# === JIT-compiled inference functions ===

@partial(jax.jit, static_argnums=(0,))
def forward_inference(
    model: TalModelJAX,
    params: Dict[str, Any],
    x: jnp.ndarray,
) -> ModelOutput:
    """
    JIT-compiled inference forward pass.
    
    Args:
        model: TalModelJAX instance.
        params: Model parameters.
        x: (B, C, H, W) or (B, H, W, C) input tensor.
        
    Returns:
        ModelOutput with value and policy.
    """
    return model.apply(params, x, train=False)


@partial(jax.jit, static_argnums=(0,))
def forward_training(
    model: TalModelJAX,
    params: Dict[str, Any],
    x: jnp.ndarray,
) -> ModelOutput:
    """
    JIT-compiled training forward pass.
    
    Args:
        model: TalModelJAX instance.
        params: Model parameters (including batch_stats).
        x: (B, C, H, W) or (B, H, W, C) input tensor.
        
    Returns:
        ModelOutput with value and policy.
    """
    return model.apply(params, x, train=True, mutable=["batch_stats"])

