# agent.py
import numpy as np
import pickle
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        # Use a nested dictionary for the Q-table: Q[state][action] = value
        # This is much more efficient for a sparse/dynamic action space.
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        return self.q_table[state_key][action]

    def choose_action(self, state, valid_actions):
        """Chooses a single action from a list of valid actions."""
        if not valid_actions:
            return None # No action is possible

        if np.random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploit: choose the best known action
        q_values = {action: self.get_q_value(state, action) for action in valid_actions}
        return max(q_values, key=q_values.get)

    def update(self, state, action, reward, next_state, next_valid_actions):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]

        # Find max Q-value for the next state from the set of valid actions
        max_next_q = 0
        if next_valid_actions:
            max_next_q = max(self.q_table[next_state_key][a] for a in next_valid_actions)
        
        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def _get_state_key(self, state):
        """
        The state is already a tuple from the environment, so it can be used directly as a key.
        This function is kept for consistency but could be removed.
        """
        return state

    def save_q_table(self, path):
        # Convert defaultdicts to regular dicts for safer pickling
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump(q_table_dict, f)

    def load_q_table(self, path):
        with open(path, "rb") as f:
            q_table_dict = pickle.load(f)
        # Rebuild the nested defaultdict structure
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in q_table_dict.items():
            for action, value in actions.items():
                self.q_table[state][action] = value