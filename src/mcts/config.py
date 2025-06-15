"""
Configuration for the MCTS engine.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    """
    A single configuration object for the entire MCTS system.

    This dataclass centralizes all parameters, making it easier to manage
    and pass around configuration for controllers, workers, and managers.
    """

    # --- Controller Configuration ---
    num_workers: int = 4
    model_weights_path: Optional[str] = None
    buffer_count: int = 256  # Number of shared memory buffers

    # --- Worker Configuration ---
    c_puct: float = 1.0  # Exploration-exploitation trade-off in PUCT
    n_scl: int = 1000  # Node visit count scaling factor

    # --- Manager Configuration ---
    batch_size: int = 32
    max_wait_time_ms: float = 10.0

    # --- Debugging and Testing ---
    use_mock_model: bool = False
    use_error_model: bool = False
    profile_memory: bool = False
