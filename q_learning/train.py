from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
import networkx as nx
import pandas as pd
import joblib

import pandas as pd
import networkx as nx
import random

def create_small_cpn():
    """Create a small CPN with 5 courses and manual prerequisites."""
    cpn = nx.DiGraph()
    
    # Add 5 random courses (replace with your actual course IDs)
    courses = ["CSCI101", "MATH112", "CSCI201", "MATH201"]
    cpn.add_nodes_from(courses)
    
    # Define prerequisites manually
    prerequisites = {
        "CSCI101": [],          # No prerequisites
        "MATH112": [],          # No prerequisites
        "CSCI201": ["CSCI101"], # Requires CSCI101
        "MATH201": ["MATH112"], # Requires MATH112
    }
    
    # Add edges to the graph
    for course, prereqs in prerequisites.items():
        for prereq in prereqs:
            cpn.add_edge(prereq, course)
    
    return cpn

def train_agent(env, episodes=1000):
    # Get all possible actions (all courses)
    action_space = list(env.cpn.nodes())
    
    # Initialize agent
    agent = QLearningAgent(
        state_space=None,  # Our state space is dynamic
        action_space=action_space,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2
    )
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()
            
            agent.update(state, action, reward, next_state, next_valid_actions)
            
            state = next_state
            total_reward += reward
            
            # Optional: print progress
            if done or len(env.state) % 5 == 0:
                print(f"Course: {action}, Predicted Grade: {info.get('predicted_grade', 'N/A'):.2f}, Reward: {reward:.2f}")
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Final GPA: {env.current_gpa:.2f}")
    
    return agent

if __name__ == "__main__":
    # Load grade prediction model
    grade_predictor = joblib.load('helper/poly_reg_pipeline.pkl')
    
    # Load course features
    course_features_df = pd.read_csv('helper/course_features.csv')
    
    # Sample student profile
    student_profile = {
        'current_gpa': 3.2,
        'interests': ['AI', 'Systems'],
        'completed_courses': [],
        'fail_rate': 0.05  # Student's historical fail rate
    }
    
    priorities = {
        'focus': 'balanced',  # or 'gpa', 'speed', 'interests'
        'min_gpa': 3.0
    }
    
    # Create environment
    cpn = create_small_cpn()
    env = CurriculumEnvironment(
        cpn=cpn,
        student_profile=student_profile,
        priorities=priorities,
        course_features_df=course_features_df,
        grade_predictor=grade_predictor
    )
    
    agent = train_agent(env, episodes=1000)
    
    agent.save_q_table("academic_advisor_qtable.pkl")