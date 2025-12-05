"""
Flax/JAX implementation of the Tal Model (HybridChessModel).

Matches the PyTorch architecture for weight parity.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import unfreeze
from src.utils import sentinel

logger = logging.getLogger(__name__)

ACTION_SPACE_SIZE = 4672


class ModelOutput(NamedTuple):
    value: jnp.ndarray
    policy_logits: jnp.ndarray


class ResidualBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]
        residual = x

        x = nn.Conv(self.out_channels, (3, 3), padding="SAME", use_bias=False, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.silu(x)

        x = nn.Conv(self.out_channels, (3, 3), padding="SAME", use_bias=False, name="conv2")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)

        if in_channels != self.out_channels:
            residual = nn.Conv(self.out_channels, (1, 1), use_bias=False, name="projection")(residual)

        return nn.silu(x + residual)


class TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not train,
            use_bias=True,
            name="attention",
        )(x, x)
        attn = nn.Dropout(self.dropout, deterministic=not train)(attn)
        x = nn.LayerNorm(epsilon=1e-6, name="norm1")(residual + attn)

        residual = x
        y = nn.Dense(self.ff_dim, name="ffn_1")(x)
        y = nn.silu(y)
        y = nn.Dropout(self.dropout, deterministic=not train)(y)
        y = nn.Dense(self.embed_dim, name="ffn_2")(y)

        x = nn.LayerNorm(epsilon=1e-6, name="norm2")(residual + y)
        return x


class TalModelJAX(nn.Module):
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
    def __call__(self, x: jnp.ndarray, train: bool = True) -> ModelOutput:
        batch_size = x.shape[0]
        embed_dim = self.num_heads * self.key_dim

        if x.shape[1] == self.input_channels:
            x = jnp.transpose(x, (0, 2, 3, 1))

        x = nn.Conv(self.stem_filters[0], (3, 3), padding="SAME", use_bias=False, name="initial_conv")(x)
        x = nn.BatchNorm(use_running_average=not train, name="initial_bn")(x)
        x = nn.silu(x)

        for i, filters in enumerate(self.stem_filters):
            x = ResidualBlock(filters, name=f"residual_block_{i}")(x, train=train)

        x = x.reshape(batch_size, 64, embed_dim)
        for i in range(self.num_transformer_layers):
            x = TransformerEncoderBlock(
                embed_dim, self.num_heads, self.ff_dim, self.dropout, name=f"transformer_block_{i}"
            )(x, train=train)

        val = jnp.mean(x, axis=1)
        val = nn.Dense(256, name="value_fc1")(val)
        val = nn.silu(val)
        val = nn.Dense(3, name="value_fc2")(val)
        value_probs = nn.softmax(val, axis=-1)

        pol = x.reshape(batch_size, 8, 8, embed_dim)
        pol = nn.Conv(2, (1, 1), name="policy_conv")(pol)
        pol = nn.silu(pol)
        pol = pol.reshape(batch_size, -1)
        policy_logits = nn.Dense(self.action_space_size, name="policy_fc")(pol)

        return ModelOutput(value_probs, policy_logits)

    @classmethod
    def from_pytorch(cls, weights_path: str, **kwargs) -> Tuple["TalModelJAX", Dict[str, Any]]:
        from src.training_ppo.models.jax_bridge import load_pytorch_weights_to_flax

        model = cls(**kwargs)
        init_vars = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 34)))
        pt_vars = load_pytorch_weights_to_flax(weights_path, model)

        def merge(base, new):
            for k, v in new.items():
                if isinstance(v, dict) and k in base:
                    merge(base[k], v)
                else:
                    base[k] = v
            return base

        params = merge(unfreeze(init_vars["params"]), pt_vars["params"])
        batch_stats = merge(unfreeze(init_vars.get("batch_stats", {})), pt_vars.get("batch_stats", {}))

        return model, {"params": params, "batch_stats": batch_stats}


def create_model(config=None) -> TalModelJAX:
    return TalModelJAX(**(config or {}))

