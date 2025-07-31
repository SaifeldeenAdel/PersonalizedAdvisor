import numpy as np
from agent import QLearningAgent
from reward import compute_reward
from graph_utils import get_valid_actions  # Assume this exists to check prerequisites


def train_agent(env, episodes=1000):
    # Initialize agent
    agent = QLearningAgent(
        state_space=env.state_space,
        action_space=env.action_space,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
    )

    for episode in range(episodes):
        state = env.reset()  # Initial state (empty transcript)
        valid_actions = get_valid_actions(state, env.cpn)
        done = False

        while not done:
            # Select action
            action = agent.choose_action(state, valid_actions)

            # Take action (simulate semester)
            next_state, done = env.step(action)
            next_valid_actions = get_valid_actions(next_state, env.cpn)

            # Calculate reward
            course_features = env.get_course_features(action)
            reward = compute_reward(
                course_features, env.student_profile, env.priorities
            )

            # Update agent
            agent.update(state, action, reward, next_state, next_valid_actions)

            # Update state
            state = next_state
            valid_actions = next_valid_actions

        # Decay epsilon for less exploration over time
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.2f}")

    return agent
