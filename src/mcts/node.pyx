# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import cython
import chess
import numpy as np
from typing import Dict, Optional

from libc.math cimport sqrt

cdef class MCTSNode:
    cdef public object parent
    cdef public dict children
    cdef public int visit_count
    cdef public float mean_action_value
    cdef public float prior_probability
    cdef public int depth
    cdef public bint is_frozen
    cdef public dict frozen_visit_counts

    def __cinit__(self, object parent=None, float prior_p=1.0, int depth=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.mean_action_value = 0.0
        self.prior_probability = prior_p
        self.depth = depth
        self.is_frozen = False
        self.frozen_visit_counts = None

    cpdef float q_value(self):
        return self.mean_action_value

    cpdef bint is_leaf(self):
        return not self.children

    cpdef expand(self, dict policy_output):
        cdef float probability
        cdef object move
        for move, probability in policy_output.items():
            if move not in self.children:
                self.children[move] = MCTSNode(
                    parent=self, prior_p=probability, depth=self.depth + 1
                )

    cpdef update(self, float value):
        self.visit_count += 1
        self.mean_action_value += (value - self.mean_action_value) / self.visit_count
        if self.parent:
            (<MCTSNode>self.parent).update(-value)

    cpdef select_child(self, float c_puct, int n_scl):
        # C-level local variables must be declared before any Python statements in the
        #   function scope.  Declare them once here and use them later in the logic.
        cdef int current_total_visits
        cdef MCTSNode child
        cdef object move

        if not self.children:
            return None

        if self.depth % 2 != 0:
            if self.is_frozen:
                return self._thompson_sample()

            # --- MODIFICATION: Replaced generator expression with an explicit for loop ---
            current_total_visits = 0
            for child in self.children.values():
                current_total_visits += (<MCTSNode>child).visit_count
            # --- END OF MODIFICATION ---

            if current_total_visits > n_scl:
                # --- MODIFICATION: Replaced dict comprehension with an explicit for loop ---
                frozen_counts = {}
                for move, child in self.children.items():
                    if (<MCTSNode>child).visit_count > 0:
                        frozen_counts[move] = (<MCTSNode>child).visit_count
                self.frozen_visit_counts = frozen_counts
                # --- END OF MODIFICATION ---

                self.is_frozen = True
                return self._thompson_sample()

        return self._select_best_child_puct(c_puct)

    cpdef _thompson_sample(self):
        if not self.frozen_visit_counts:
            return self._select_best_child_puct(0.0)

        moves = list(self.frozen_visit_counts.keys())
        counts = np.array(list(self.frozen_visit_counts.values()), dtype=np.float32)
        probabilities = counts / counts.sum()
        chosen_move = np.random.choice(moves, p=probabilities)
        return chosen_move, self.children[chosen_move]

    cpdef _select_best_child_puct(self, float c_puct):
        cdef float best_score = -99999.0
        cdef float q_value, score, exploration_term
        cdef MCTSNode best_child = None
        cdef MCTSNode child
        cdef object best_move = None
        cdef object move
        cdef float sqrt_total_visits = sqrt(self.visit_count)

        for move, child_obj in self.children.items():
            child = <MCTSNode>child_obj
            q_value = child.mean_action_value
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