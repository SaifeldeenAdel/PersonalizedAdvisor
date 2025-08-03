from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
import networkx as nx

# SAMPLE COURSE PREREQUISITE NETWORK
def create_sample_cpn():
    """Create a sample course prerequisite network"""
    cpn = nx.DiGraph()
    
    courses = [f"CS{i}" for i in range(100, 400)] + [f"MATH{i}" for i in range(100, 300)]
    cpn.add_nodes_from(courses)
    
    # Add some prerequisite relationships
    for i in range(200, 300):
        cpn.add_edge(f"CS{i}", f"CS{i+100}")
        cpn.add_edge(f"MATH{i}", f"CS{i+100}")
    
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
            if not valid_actions:  # No valid actions (shouldn't happen if graph is correct)
                break
                
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            next_valid_actions = env.get_valid_actions()
            
            agent.update(state, action, reward, next_state, next_valid_actions)
            
            state = next_state
            total_reward += reward
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return agent

if __name__ == "__main__":
    # Sample student profile
    student_profile = {
        'current_gpa': 3.2,
        'interests': ['AI', 'Systems'],
        'completed_courses': []
    }
    
    priorities = {
        'focus': 'balanced',  # or 'gpa', 'speed', 'interests'
        'min_gpa': 3.0
    }
    
    # Create environment
    cpn = create_sample_cpn()
    print(cpn.nodes())
    env = CurriculumEnvironment(cpn, student_profile, priorities)
    
    agent = train_agent(env, episodes=1000)
    
    agent.save_q_table("academic_advisor_qtable.pkl")