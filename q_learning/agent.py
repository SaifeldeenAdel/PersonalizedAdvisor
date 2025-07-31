class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.2):
        pass

    def get_q_value(self, state, action):
        pass

    def choose_action(self, state, valid_actions):
        pass

    def update(self, state, action, reward, next_state, next_valid_actions):
        pass

    def save_q_table(self, path):
        pass

    def load_q_table(self, path):
        pass
