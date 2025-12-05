"""
Simple single-process MCTS for PyTorch models.

This is a lightweight MCTS implementation that:
- Uses the PyTorch TransformerEvaluator directly (no multiprocess overhead)
- Runs entirely in-process (no shared memory IPC complexity)
- Is suitable for experiments and evaluation, not high-throughput training

Trade-offs:
- Simpler but slower than the parallel implementation
- No TensorFlow dependency issues
- Good for testing Coach Tal integration with MCTS
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

from src.coach_tal.evaluator import TransformerEvaluator

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.
    
    Attributes:
        prior: Prior probability from the policy network.
        visit_count: Number of times this node has been visited.
        value_sum: Sum of all values backpropagated through this node.
        children: Dict mapping moves to child nodes.
    """
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    
    def is_expanded(self) -> bool:
        """Returns True if this node has been expanded (has children)."""
        return len(self.children) > 0
    
    def q_value(self) -> float:
        """Returns the mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float = 4.0) -> float:
        """
        Compute the UCB score for node selection.
        
        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        
        Args:
            parent_visit_count: Visit count of the parent node.
            c_puct: Exploration constant.
            
        Returns:
            UCB score for this node.
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration


class SimpleMCTS:
    """
    Simple single-process MCTS using PyTorch TransformerEvaluator.
    
    This implementation is suitable for experiments and evaluation.
    For high-throughput training, use the parallel implementation.
    
    Example:
        evaluator = TransformerEvaluator(weights_path="model.pt", use_pytorch=True)
        mcts = SimpleMCTS(evaluator, c_puct=4.0)
        policy, q_value = mcts.search(board, num_simulations=800)
    """
    
    def __init__(
        self,
        evaluator: TransformerEvaluator,
        c_puct: float = 4.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        add_noise: bool = False,
    ):
        """
        Initialize SimpleMCTS.
        
        Args:
            evaluator: TransformerEvaluator for position evaluation.
            c_puct: Exploration constant for UCB.
            dirichlet_alpha: Alpha for Dirichlet noise (exploration).
            dirichlet_epsilon: Epsilon for mixing Dirichlet noise.
            add_noise: Whether to add Dirichlet noise at root (for training).
        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise
    
    def search(
        self,
        board: chess.Board,
        num_simulations: int,
    ) -> Tuple[Dict[chess.Move, float], float]:
        """
        Run MCTS search from a position.
        
        Args:
            board: The chess position to search from.
            num_simulations: Number of simulations to run.
            
        Returns:
            Tuple of:
                - policy: Dict mapping moves to visit proportions (sums to 1).
                - q_value: The root Q value (expected outcome).
        """
        root = MCTSNode()
        
        # Expand root node
        value, policy = self.evaluator.evaluate(board)
        self._expand_node(root, policy)
        
        # Add Dirichlet noise at root for exploration (if enabled)
        if self.add_noise and root.children:
            self._add_dirichlet_noise(root)
        
        # Backpropagate initial value
        root.visit_count += 1
        root.value_sum += value
        
        # Run simulations
        for _ in range(num_simulations - 1):  # -1 because we already did root expansion
            self._run_simulation(root, board)
        
        # Compute final policy from visit counts
        final_policy = self._get_policy_from_visits(root)
        q_value = root.q_value()
        
        return final_policy, q_value
    
    def _run_simulation(self, root: MCTSNode, board: chess.Board) -> None:
        """Run a single MCTS simulation."""
        node = root
        search_board = board.copy()
        path: List[MCTSNode] = [node]
        
        # Selection: traverse tree using UCB until we reach an unexpanded node
        while node.is_expanded():
            move, child = self._select_child(node)
            search_board.push(move)
            node = child
            path.append(node)
        
        # Handle terminal states
        if search_board.is_game_over(claim_draw=True):
            value = self._get_terminal_value(search_board)
        else:
            # Expansion and evaluation
            value, policy = self.evaluator.evaluate(search_board)
            self._expand_node(node, policy)
            # Value is from perspective of player to move, negate for parent
            value = -value
        
        # Backpropagation
        self._backpropagate(path, value)
    
    def _select_child(self, node: MCTSNode) -> Tuple[chess.Move, MCTSNode]:
        """Select the child with highest UCB score."""
        best_score = float('-inf')
        best_move = None
        best_child = None
        
        for move, child in node.children.items():
            score = child.ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def _expand_node(self, node: MCTSNode, policy: Dict[chess.Move, float]) -> None:
        """Expand a node with children for each legal move."""
        for move, prob in policy.items():
            node.children[move] = MCTSNode(prior=prob)
    
    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        """Backpropagate value through the path, alternating signs."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective
    
    def _get_terminal_value(self, board: chess.Board) -> float:
        """Get the value of a terminal position."""
        if board.is_checkmate():
            return -1.0  # Current player is mated
        return 0.0  # Draw
    
    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root node priors for exploration."""
        moves = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        
        for move, n in zip(moves, noise):
            child = node.children[move]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n
    
    def _get_policy_from_visits(self, root: MCTSNode) -> Dict[chess.Move, float]:
        """Convert visit counts to a policy distribution."""
        if not root.children:
            return {}
        
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            # Uniform over children
            uniform = 1.0 / len(root.children)
            return {move: uniform for move in root.children}
        
        return {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }
    
    def get_raw_visit_counts(self, root: MCTSNode) -> Dict[chess.Move, int]:
        """Get raw visit counts (useful for aggregation in parallel search)."""
        return {move: child.visit_count for move, child in root.children.items()}


@dataclass
class SimpleMCTSResult:
    """Result from SimpleMCTS search."""
    policy: Dict[chess.Move, float]
    q_value: float
    error: Optional[str] = None


class SimpleMCTSController:
    """
    Controller that manages SimpleMCTS with a cached evaluator.
    
    This provides a similar interface to MCTSController but uses
    the simpler in-process SimpleMCTS implementation.
    """
    
    def __init__(
        self,
        model_weights_path: str,
        use_pytorch: bool = True,
        c_puct: float = 4.0,
        add_noise: bool = False,
    ):
        """
        Initialize the controller.
        
        Args:
            model_weights_path: Path to model weights.
            use_pytorch: Use PyTorch backend (recommended).
            c_puct: Exploration constant for UCB.
            add_noise: Whether to add exploration noise at root.
        """
        self.model_weights_path = model_weights_path
        self.use_pytorch = use_pytorch
        self.c_puct = c_puct
        self.add_noise = add_noise
        
        self._evaluator: Optional[TransformerEvaluator] = None
        self._mcts: Optional[SimpleMCTS] = None
        self._is_started = False
    
    def start(self) -> None:
        """Initialize the evaluator and MCTS."""
        if self._is_started:
            return
        
        self._evaluator = TransformerEvaluator(
            weights_path=self.model_weights_path,
            use_pytorch=self.use_pytorch,
            temperature=1.0,
        )
        # Force initialization
        self._evaluator._ensure_initialized()
        
        self._mcts = SimpleMCTS(
            evaluator=self._evaluator,
            c_puct=self.c_puct,
            add_noise=self.add_noise,
        )
        
        self._is_started = True
        logger.info(f"SimpleMCTSController started with model from {self.model_weights_path}")
    
    def run_search(self, fen: str, num_simulations: int) -> SimpleMCTSResult:
        """
        Run MCTS search on a position.
        
        Args:
            fen: FEN string of the position.
            num_simulations: Number of simulations to run.
            
        Returns:
            SimpleMCTSResult with policy and Q value.
        """
        if not self._is_started:
            raise RuntimeError("SimpleMCTSController must be started before running searches")
        
        try:
            board = chess.Board(fen)
            policy, q_value = self._mcts.search(board, num_simulations)
            
            # Convert Move objects to UCI strings for compatibility
            policy_str = {
                (move.uci() if isinstance(move, chess.Move) else move): prob
                for move, prob in policy.items()
            }
            
            return SimpleMCTSResult(policy=policy_str, q_value=q_value)
        except Exception as e:
            logger.error(f"Error in SimpleMCTS search: {e}")
            return SimpleMCTSResult(policy={}, q_value=0.0, error=str(e))
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self._evaluator = None
        self._mcts = None
        self._is_started = False
        logger.info("SimpleMCTSController shut down")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


