import chess
import math
import numpy as np
from typing import Dict, Optional


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.

    This class encapsulates all the logic and state required for both
    standard MCTS and the advanced Search-Contempt algorithm.
    """

    def __init__(
        self, parent: Optional["MCTSNode"] = None, prior_p: float = 1.0, depth: int = 0
    ):
        """
        Initializes a new MCTSNode.

        Args:
            parent: The parent node of this node.
            prior_p: The prior probability of selecting this node from its parent.
            depth: The depth of the node in the search tree.
        """
        self.parent = parent
        self.children: Dict[chess.Move, "MCTSNode"] = {}

        # Core MCTS statistics
        self.visit_count: int = 0
        self.mean_action_value: float = 0.0  # Q-value
        self.prior_probability: float = prior_p

        # Search-Contempt attributes
        self.depth: int = depth
        self.is_frozen: bool = False
        self.frozen_visit_counts: Optional[Dict[chess.Move, int]] = None

    @property
    def q_value(self) -> float:
        """
        Returns the mean action value (Q) of this node.

        Returns 0 if the node has not been visited to avoid division by zero.
        """
        return self.mean_action_value

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (i.e., has no children)."""
        return not self.children

    def expand(self, policy_output: Dict[chess.Move, float]):
        """
        Expands the node by creating children for all legal actions.

        This is called when the MCTS search reaches an unvisited leaf node.

        Args:
            policy_output: A dictionary mapping legal moves to their prior
                           probabilities from the neural network's policy head.
        """
        for move, probability in policy_output.items():
            if move not in self.children:
                self.children[move] = MCTSNode(
                    parent=self, prior_p=probability, depth=self.depth + 1
                )

    def update(self, value: float):
        """
        Propagates the result of a simulation (the value) back up the tree.

        This method updates the node's statistics and recursively calls the
        parent's update method.

        Args:
            value: The value from the simulation, from the perspective of the
                   current node's player.
        """
        self.visit_count += 1
        # Update the mean action value using a running average formula
        self.mean_action_value += (value - self.mean_action_value) / self.visit_count

        if self.parent:
            # The value must be inverted for the parent, as it represents the
            # opponent's perspective.
            self.parent.update(-value)

    def select_child(
        self, c_puct: float, n_scl: int
    ) -> Optional[tuple[chess.Move, "MCTSNode"]]:
        """
        Selects a child node using the asymmetric Search-Contempt logic.

        - For "player" nodes (even depth), it uses the PUCT formula.
        - For "opponent" nodes (odd depth), it uses PUCT until a visit
          threshold (n_scl) is met, then switches to Thompson Sampling from a
          "frozen" visit distribution.

        Args:
            c_puct: The exploration constant for the PUCT formula.
            n_scl: The visit count threshold for the Search-Contempt logic.

        Returns:
            A tuple of (move, child_node) for the selected child, or None if no children.
        """
        if not self.children:
            return None

        # Asymmetric Logic: Check for "opponent node"
        if self.depth % 2 != 0:
            # Thompson Sampling for frozen opponent nodes
            if self.is_frozen:
                return self._thompson_sample()

            # Check if the node should be frozen
            current_total_visits = sum(
                child.visit_count for child in self.children.values()
            )
            if current_total_visits > n_scl:
                self.frozen_visit_counts = {
                    move: child.visit_count
                    for move, child in self.children.items()
                    if child.visit_count > 0
                }
                self.is_frozen = True
                return self._thompson_sample()

        # Default case: Player node or unfrozen opponent node uses PUCT
        return self._select_best_child_puct(c_puct)

    def _thompson_sample(self) -> Optional[tuple[chess.Move, "MCTSNode"]]:
        """Performs Thompson Sampling from the frozen visit distribution."""
        if not self.frozen_visit_counts:
            # This can happen if freezing occurs but no children had any visits.
            # Fallback to PUCT selection among all children.
            return self._select_best_child_puct(0.0)  # c_puct=0 to pick best Q

        moves = list(self.frozen_visit_counts.keys())
        counts = np.array(list(self.frozen_visit_counts.values()), dtype=np.float32)
        probabilities = counts / counts.sum()

        chosen_move = np.random.choice(moves, p=probabilities)
        return chosen_move, self.children[chosen_move]

    def _select_best_child_puct(
        self, c_puct: float
    ) -> Optional[tuple[chess.Move, "MCTSNode"]]:
        """Selects the best child using the PUCT formula."""
        best_score = -float("inf")
        best_move = None
        best_child = None

        sqrt_total_visits = math.sqrt(self.visit_count)

        for move, child in self.children.items():
            # PUCT = Q(s,a) + c_puct * P(s,a) * (sqrt(N(s)) / (1 + N(s,a)))
            q_value = child.q_value
            exploration_term = (
                c_puct
                * child.prior_probability
                * (sqrt_total_visits / (1 + child.visit_count))
            )
            score = q_value + exploration_term

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_move is None:
            return None

        return best_move, best_child
