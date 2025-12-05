"""
Pure-Python fallback for the Cython `node` extension.

If the compiled extension `node` (node.cpython-*.so) is unavailable or built
for a different Python version, this module provides a minimal implementation
of `MCTSNode` used by tests and higher-level MCTS code. It matches the public
interface expected by tests in `tests/test_node.py`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import chess


class MCTSNode:
    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior_p: float = 1.0,
        depth: int = 0,
    ):
        self.parent = parent
        self.prior_probability = prior_p
        self.depth = depth

        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visit_count: int = 0
        self.mean_action_value: float = 0.0

        # Freezing state for opponent nodes (Thompson sampling)
        self.is_frozen: bool = False
        self.frozen_visit_counts: Optional[Dict[chess.Move, int]] = None

    # Property alias used in tests
    @property
    def q_value(self) -> float:
        return self.mean_action_value

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(self, policy: Dict[chess.Move, float]) -> None:
        for move, prob in policy.items():
            if move not in self.children:
                self.children[move] = MCTSNode(parent=self, prior_p=prob, depth=self.depth + 1)

    def update(self, value: float) -> None:
        # Standard incremental mean update
        self.visit_count += 1
        self.mean_action_value += (value - self.mean_action_value) / self.visit_count

        # Propagate to parent with perspective flip
        if self.parent is not None:
            self.parent.update(-value)

    def select_child(self, c_puct: float, n_scl: int):
        if self.is_leaf():
            return None

        # If already frozen, sample based on stored counts
        if self.is_frozen and self.frozen_visit_counts:
            moves = list(self.frozen_visit_counts.keys())
            counts = np.array(list(self.frozen_visit_counts.values()), dtype=np.float32)
            probs = counts / counts.sum()
            chosen = np.random.choice(moves, p=probs)
            return chosen, self.children[chosen]

        total_visits = sum(child.visit_count for child in self.children.values())
        # Freeze when total visits exceed threshold
        if total_visits > n_scl:
            self.is_frozen = True
            self.frozen_visit_counts = {m: c.visit_count for m, c in self.children.items()}
            moves = list(self.frozen_visit_counts.keys())
            counts = np.array(list(self.frozen_visit_counts.values()), dtype=np.float32)
            probs = counts / counts.sum()
            chosen = np.random.choice(moves, p=probs)
            return chosen, self.children[chosen]

        # PUCT selection
        best_move = None
        best_score = -math.inf
        parent_visits = self.visit_count
        for move, child in self.children.items():
            prior = child.prior_probability
            q = child.q_value
            u = c_puct * prior * (math.sqrt(parent_visits) / (1 + child.visit_count)) if parent_visits > 0 else 0.0
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            return None
        return best_move, self.children[best_move]


__all__ = ["MCTSNode"]

