"""
Centralized specifications and constants for Tal-RL.

This module defines all shared constants to avoid magic numbers
scattered across the codebase.
"""

# =============================================================================
# Action Space
# =============================================================================

# Total possible move encoding in AlphaZero-style representation:
# 73 move types per square × 64 squares = 4672
# Move types: 56 queen-like + 8 knight + 9 underpromotions
ACTION_SPACE_SIZE = 4672

# =============================================================================
# Board Encoding
# =============================================================================

# Board dimensions
BOARD_SIZE = 8

# Tal model input channels:
# 0-11: Piece positions (12)
# 12-15: Castling rights (4)
# 16: Material score (1)
# 17: En passant (1)
# 18-19: Move counters (2)
# 20: Mobility (1)
# 21-23: Pawn structure (3)
# 24-25: Control/PST (2)
# 26-29: Defended/vulnerable (4)
# 30-31: Coordination (2)
# 32: Game phase (1)
# 33: King safety (1)
# Total: 34 channels
INPUT_CHANNELS = 34

# =============================================================================
# Model Architecture
# =============================================================================

STEM_FILTERS = (128, 128, 256, 256)
TRANSFORMER_LAYERS = 6
ATTENTION_HEADS = 8
KEY_DIM = 32
EMBED_DIM = ATTENTION_HEADS * KEY_DIM  # 256
FF_DIM = 1024
DROPOUT = 0.1

# =============================================================================
# Players
# =============================================================================

WHITE = 0
BLACK = 1
AGENT = WHITE   # Agent plays White
VICTIM = BLACK  # Victim plays Black

