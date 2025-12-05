"""
Lightweight transformer evaluator wrapper for Coach Tal.

This module provides a clean interface for neural network inference,
returning (value, policy_dict) for arbitrary board positions. It handles
encoding and masking internally.

Note: The model was trained to handle both White and Black perspectives
directly (no board mirroring), outputting values from the current player's
perspective. Inference matches this: positions are encoded as-is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import chess
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransformerEvaluator:
    """
    Wraps the transformer model for position evaluation.
    
    Provides a unified interface that:
        - Loads weights once and caches the model
        - Handles board encoding (HWC format for TF, CHW for PyTorch)
        - Masks illegal moves and normalizes policy to legal moves only
        - Returns value as a scalar and policy as Dict[Move, float]
    
    Attributes:
        weights_path: Path to model weights (.keras or .pt file).
        use_pytorch: If True, use PyTorch model; otherwise use TensorFlow/Keras.
        temperature: Softmax temperature for policy (1.0 = no change).
    """
    
    weights_path: str
    use_pytorch: bool = False
    temperature: float = 1.0
    
    # Private fields initialized in __post_init__
    _model: object = field(default=None, init=False, repr=False)
    _encoder: object = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Lazy initialization – model is loaded on first evaluate() call."""
        pass
    
    def _ensure_initialized(self) -> None:
        """Load model and encoder if not already done."""
        if self._initialized:
            return
        
        from src.encoder import Encoder
        from src.move_mapping import move_to_index, ACTION_SPACE_SIZE
        
        self._encoder = Encoder()
        self._action_space_size = ACTION_SPACE_SIZE
        self._move_to_index = move_to_index
        
        if self.use_pytorch:
            self._load_pytorch_model()
        else:
            self._load_keras_model()
        
        self._initialized = True
        logger.info(f"TransformerEvaluator initialized with weights from {self.weights_path}")
    
    def _load_keras_model(self) -> None:
        """Load TensorFlow/Keras model."""
        import tensorflow as tf
        
        # Configure GPU memory growth
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Already configured
        
        # Try loading as full model first, fall back to architecture + weights
        try:
            self._model = tf.keras.models.load_model(
                self.weights_path,
                custom_objects={"swish": tf.nn.silu},
                compile=False,
            )
            logger.info("Loaded full Keras model")
        except Exception:
            # Fall back to creating architecture and loading weights
            from src.transformer_model import create_model
            self._model = create_model()
            self._model.load_weights(self.weights_path)
            logger.info("Loaded Keras model architecture + weights")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        import torch
        from src.transformer_model_pytorch import HybridChessModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        
        self._model = HybridChessModel()
        checkpoint = torch.load(self.weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._model.eval()
        logger.info(f"Loaded PyTorch model on {device}")
    
    def evaluate(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """
        Evaluate a board position.
        
        Args:
            board: The chess position to evaluate.
            
        Returns:
            Tuple of:
                - value: Scalar evaluation in [-1, 1] from current player's perspective.
                        +1 = winning, -1 = losing, 0 = drawn.
                - policy: Dict mapping legal moves to probabilities (sums to 1).
        """
        self._ensure_initialized()
        
        # Handle terminal positions
        if board.is_game_over(claim_draw=True):
            return self._handle_terminal(board), {}
        
        # Encode and run inference
        if self.use_pytorch:
            value, logits = self._infer_pytorch(board)
        else:
            value, logits = self._infer_keras(board)
        
        # Convert logits to legal-move policy dict
        policy = self._logits_to_policy(logits, board)
        
        return value, policy
    
    def _infer_keras(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """Run inference - model handles both perspectives directly (no mirroring)."""
        # Encode board directly - NO mirroring (matches training)
        encoded = self._encoder.encode(board)
        tensor = np.expand_dims(encoded, axis=0)
        
        # Predict
        value_output, policy_logits = self._model.predict(tensor, verbose=0)
        
        # Extract value (model outputs from current player's perspective)
        if value_output.shape[-1] == 3:
            # WDL format: [loss, draw, win] - compute expected value: win - loss
            value = float(value_output[0, 2] - value_output[0, 0])
        else:
            value = float(value_output[0, 0])
        
        # Policy logits are already in correct format (no unmirroring needed)
        logits = policy_logits[0]
        
        return value, logits
    
    def _infer_pytorch(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """Run inference - model handles both perspectives directly (no mirroring)."""
        import torch
        
        # Encode board directly - NO mirroring (matches training)
        encoded = self._encoder.encode(board)
        encoded_chw = np.transpose(encoded, (2, 0, 1))
        # Make contiguous copy to handle negative strides from np.flip in encoder
        encoded_chw = np.ascontiguousarray(encoded_chw)
        tensor = torch.from_numpy(encoded_chw).unsqueeze(0).float().to(self._device)
        
        with torch.no_grad():
            value_output, policy_logits = self._model(tensor)
        
        # Extract value (model outputs from current player's perspective)
        value_np = value_output.cpu().numpy()
        if value_np.shape[-1] == 3:
            value = float(value_np[0, 2] - value_np[0, 0])
        else:
            value = float(value_np[0, 0])
        
        # Policy logits are already in correct format (no unmirroring needed)
        logits = policy_logits.cpu().numpy()[0]
        
        return value, logits
    
    def _logits_to_policy(
        self, logits: np.ndarray, board: chess.Board
    ) -> Dict[chess.Move, float]:
        """
        Convert raw policy logits to a probability dict over legal moves.
        
        Applies masking to illegal moves and softmax with temperature.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}
        
        # Get indices for legal moves
        legal_indices = []
        index_to_move = {}
        for move in legal_moves:
            idx = self._move_to_index(move, board)
            if idx is not None:
                legal_indices.append(idx)
                index_to_move[idx] = move
        
        if not legal_indices:
            # Fallback: uniform over legal moves
            uniform_prob = 1.0 / len(legal_moves)
            return {m: uniform_prob for m in legal_moves}
        
        # Extract logits for legal moves only
        legal_logits = logits[legal_indices]
        
        # Apply temperature
        if self.temperature != 1.0:
            legal_logits = legal_logits / self.temperature
        
        # Stable softmax
        max_logit = np.max(legal_logits)
        exp_logits = np.exp(legal_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        # Build policy dict
        policy = {}
        for i, idx in enumerate(legal_indices):
            policy[index_to_move[idx]] = float(probs[i])
        
        return policy
    
    def _handle_terminal(self, board: chess.Board) -> float:
        """Return value for terminal positions."""
        if board.is_checkmate():
            return -1.0  # Current player is mated
        return 0.0  # Draw
    
    def get_raw_policy_logits(self, board: chess.Board) -> np.ndarray:
        """
        Get raw policy logits (before masking) for a position.
        
        Useful for debugging or advanced analysis.
        """
        self._ensure_initialized()
        
        if self.use_pytorch:
            _, logits = self._infer_pytorch(board)
        else:
            _, logits = self._infer_keras(board)
        
        return logits

