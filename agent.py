import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.action_space = action_space
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))

    def get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        action_idx = self.action_space.index(action)
        return self.q_table[state_key][action_idx]

    def choose_action(self, state, valid_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        state_key = self._get_state_key(state)
        q_values = {action: self.get_q_value(state, action) for action in valid_actions}
        return max(q_values.items(), key=lambda x: x[1])[0]

    def update(self, state, action, reward, next_state, next_valid_actions):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        action_idx = self.action_space.index(action)

        current_q = self.q_table[state_key][action_idx]

        if next_valid_actions:
            max_next_q = max(
                [self.get_q_value(next_state, a) for a in next_valid_actions]
            )
        else:
            max_next_q = 0  # terminal state

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

    def _get_state_key(self, state):
        """Convert state to immutable key for Q-table"""
        return tuple(sorted(state))

    def save_q_table(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, path):
        with open(path, "rb") as f:
            q_table = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), q_table)