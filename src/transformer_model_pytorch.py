"""
PyTorch implementation of the Hybrid CNN-Transformer Chess Model.

This is a faithful port of the TensorFlow/Keras model from transformer_model.py.
Key architectural components:
    - CNN Stem: 4 residual blocks extracting spatial features
    - Transformer Body: 6 encoder layers for global reasoning
    - Dual Heads: Policy (4672 actions) and Value (Win/Draw/Loss)

CRITICAL: PyTorch uses NCHW format (Batch, Channels, Height, Width)
         Input shape: (B, 34, 8, 8) instead of TensorFlow's (B, 8, 8, 34)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from src.move_mapping import ACTION_SPACE_SIZE


class ResidualBlock(nn.Module):
    """
    A single residual block for the CNN stem.
    
    Structure:
        Conv2d → BatchNorm → SiLU → Conv2d → BatchNorm → (+ residual) → SiLU
    
    If input channels != output filters, a 1x1 projection is applied to the residual.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection layer for residual if dimensions don't match
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming (He) initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if self.projection is not None:
            nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.projection is not None:
            residual = self.projection(residual)
        
        out = out + residual
        out = F.silu(out)
        
        return out


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder block with Pre-LN architecture.
    
    Structure:
        MultiHeadAttention → Dropout → (+ residual) → LayerNorm
        → FFN → Dropout → (+ residual) → LayerNorm
    
    Note: This matches the Keras implementation which uses Post-LN style
          (LayerNorm after the residual connection).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input/output: (Batch, Sequence, Feature)
        )
        self.dropout1 = nn.Dropout(dropout)
        # Match Keras model which uses epsilon=1e-6 (explicitly set in transformer_model.py)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        # Match Keras model which uses epsilon=1e-6
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize FFN weights."""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class HybridChessModel(nn.Module):
    """
    Hybrid CNN-Transformer model for chess position evaluation.
    
    Architecture:
        1. CNN Stem: Extracts local spatial features from the 8x8 board
        2. Transformer Body: Global reasoning via self-attention
        3. Dual Heads:
            - Policy: Predicts move probabilities (4672 actions)
            - Value: Predicts game outcome (Win/Draw/Loss)
    
    Input: (B, 34, 8, 8) - PyTorch NCHW format
    Outputs: Tuple of (value_probs, policy_logits)
        - value_probs: (B, 3) - Softmax probabilities for Win/Draw/Loss
        - policy_logits: (B, 4672) - Raw logits for each possible action
    """
    
    def __init__(
        self,
        input_channels: int = 34,
        action_space_size: int = ACTION_SPACE_SIZE,
        stem_filters: Tuple[int, ...] = (128, 128, 256, 256),
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        key_dim: int = 32,  # embed_dim = num_heads * key_dim = 256
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.action_space_size = action_space_size
        embed_dim = num_heads * key_dim  # 256
        
        # === CNN Stem ===
        # Initial convolution: 34 → 128 channels
        self.initial_conv = nn.Conv2d(
            input_channels, stem_filters[0], kernel_size=3, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(stem_filters[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        in_ch = stem_filters[0]
        for out_ch in stem_filters:
            self.residual_blocks.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Final stem output channels (should equal embed_dim for transformer)
        self.stem_out_channels = stem_filters[-1]
        assert self.stem_out_channels == embed_dim, \
            f"Stem output ({self.stem_out_channels}) must equal transformer embed_dim ({embed_dim})"
        
        # === Transformer Body ===
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])
        
        # === Value Head ===
        # GlobalAveragePooling1D → Dense(256) → Dense(3)
        self.value_fc1 = nn.Linear(embed_dim, 256)
        self.value_fc2 = nn.Linear(256, 3)
        
        # === Policy Head ===
        # Reshape to spatial → Conv2d(2, 1x1) → Flatten → Dense(4672)
        self.policy_conv = nn.Conv2d(embed_dim, 2, kernel_size=1, bias=True)
        self.policy_fc = nn.Linear(2 * 8 * 8, action_space_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights."""
        # Initial conv
        nn.init.kaiming_normal_(
            self.initial_conv.weight, mode='fan_out', nonlinearity='relu'
        )
        
        # Value head
        nn.init.xavier_uniform_(self.value_fc1.weight)
        nn.init.zeros_(self.value_fc1.bias)
        nn.init.xavier_uniform_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)
        
        # Policy head
        nn.init.kaiming_normal_(
            self.policy_conv.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.zeros_(self.policy_conv.bias)
        nn.init.xavier_uniform_(self.policy_fc.weight)
        nn.init.zeros_(self.policy_fc.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (B, 34, 8, 8) in NCHW format
        
        Returns:
            Tuple of (value_probs, policy_logits):
                - value_probs: (B, 3) softmax probabilities
                - policy_logits: (B, 4672) raw logits
        """
        batch_size = x.shape[0]
        
        # === 1. CNN Stem ===
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.silu(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        # x shape: (B, 256, 8, 8)
        
        # === 2. Prepare for Transformer ===
        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        # This gives us a sequence of 64 tokens, each of 256 dimensions
        x = x.flatten(2)  # (B, 256, 64)
        x = x.permute(0, 2, 1)  # (B, 64, 256) - sequence format
        
        # === 3. Transformer Body ===
        for transformer_block in self.transformer_layers:
            x = transformer_block(x)
        
        # x shape: (B, 64, 256)
        
        # === 4. Dual Heads ===
        
        # --- Value Head ---
        # Global average pooling over sequence dimension
        value_repr = x.mean(dim=1)  # (B, 256)
        value_out = F.silu(self.value_fc1(value_repr))  # (B, 256)
        value_probs = F.softmax(self.value_fc2(value_out), dim=-1)  # (B, 3)
        
        # --- Policy Head ---
        # Reshape back to spatial: (B, 64, 256) → (B, 256, 8, 8)
        policy_spatial = x.permute(0, 2, 1)  # (B, 256, 64)
        policy_spatial = policy_spatial.view(batch_size, -1, 8, 8)  # (B, 256, 8, 8)
        
        policy_out = F.silu(self.policy_conv(policy_spatial))  # (B, 2, 8, 8) in NCHW
        # CRITICAL FIX: Permute to NHWC before flattening to match Keras order
        # Keras Flatten() on (B, 8, 8, 2) gives interleaved: [r0c0ch0, r0c0ch1, r0c1ch0, ...]
        # PyTorch Flatten() on (B, 2, 8, 8) gives planar: [ch0r0c0, ch0r0c1, ..., ch1r0c0, ...]
        # Weights were trained expecting interleaved order, so we must match it
        policy_out = policy_out.permute(0, 2, 3, 1)  # (B, 8, 8, 2) - match Keras NHWC
        policy_out = policy_out.flatten(1)  # (B, 128) - now in Keras interleaved order
        policy_logits = self.policy_fc(policy_out)  # (B, 4672)
        
        return value_probs, policy_logits


def create_model() -> HybridChessModel:
    """
    Factory function to create the hybrid chess model.
    
    Returns:
        A HybridChessModel instance with default hyperparameters matching
        the TensorFlow/Keras implementation.
    """
    return HybridChessModel(
        input_channels=34,
        action_space_size=ACTION_SPACE_SIZE,
        stem_filters=(128, 128, 256, 256),
        num_transformer_layers=6,
        num_heads=8,
        key_dim=32,
        ff_dim=1024,
        dropout=0.1,
    )


# === Utility Functions for Inference ===

def prepare_input_for_pytorch(
    board_tensor: torch.Tensor,
    from_nhwc: bool = True
) -> torch.Tensor:
    """
    Prepare input tensor for PyTorch model.
    
    Args:
        board_tensor: Encoded board state
        from_nhwc: If True, expects (B, 8, 8, 34) and converts to (B, 34, 8, 8)
                   If False, expects already in NCHW format
    
    Returns:
        Tensor in PyTorch NCHW format (B, 34, 8, 8)
    """
    if from_nhwc:
        # TensorFlow NHWC → PyTorch NCHW
        # (B, H, W, C) → (B, C, H, W)
        return board_tensor.permute(0, 3, 1, 2)
    return board_tensor


# === Testing Utilities ===

def _test_model_shapes():
    """Verify model produces correct output shapes."""
    model = create_model()
    model.eval()
    
    # Test with NCHW input
    dummy_input = torch.randn(1, 34, 8, 8)
    
    with torch.no_grad():
        value_probs, policy_logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Value output shape: {value_probs.shape} (expected: [1, 3])")
    print(f"Policy output shape: {policy_logits.shape} (expected: [1, {ACTION_SPACE_SIZE}])")
    
    assert value_probs.shape == (1, 3), f"Value shape mismatch: {value_probs.shape}"
    assert policy_logits.shape == (1, ACTION_SPACE_SIZE), f"Policy shape mismatch: {policy_logits.shape}"
    
    # Verify value probabilities sum to 1
    prob_sum = value_probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, f"Value probs don't sum to 1: {prob_sum}"
    
    print("✓ All shape tests passed!")
    return True


def _count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Hybrid Chess Model - Architecture Test")
    print("=" * 60)
    
    model = create_model()
    print(f"\nModel created successfully!")
    print(f"Total trainable parameters: {_count_parameters(model):,}")
    
    print("\n" + "-" * 40)
    print("Running shape verification...")
    _test_model_shapes()
    
    print("\n" + "-" * 40)
    print("Model architecture:")
    print(model)

