from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
from cpn import build_graph, load_data
import pickle
from collections import defaultdict

def create_real_cpn():
    """Create CPN using real data"""
    catalog = load_data()
    return build_graph(catalog)

def train_agent(env, episodes=1000):
    """Train the Q-learning agent"""
    action_space = list(env.cpn.nodes())
    
    agent = QLearningAgent(
        state_space=None,
        action_space=action_space,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2
    )
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
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

def display_terminal_recommendation(cpn, student_profile, priorities, q_table_path):
    """Display recommendation path in terminal"""
    # Load Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    q_table = defaultdict(lambda: None, q_table)
    
    # Create environment
    env = CurriculumEnvironment(cpn, student_profile, priorities)
    state = env.reset()
    path = []
    credits = 0
    
    print("\n" + "="*50)
    print("ACADEMIC ADVISOR RECOMMENDATION".center(50))
    print("="*50)
    
    print("\nStudent Profile:")
    print(f"- Current GPA: {student_profile['current_gpa']}")
    print(f"- Interests: {', '.join(student_profile['interests'])}")
    print(f"- Completed Courses: {len(student_profile['completed_courses'])}")
    
    print("\nPriorities:")
    print(f"- Focus: {priorities['focus']}")
    print(f"- Minimum GPA: {priorities['min_gpa']}")
    print(f"- Preferred Tracks: {', '.join(priorities['preferred_tracks'])}")
    print(f"- Workload Preference: {priorities['workload_preference']}")
    
    print("\n" + "-"*50)
    print("RECOMMENDED COURSE SEQUENCE".center(50))
    print("-"*50)
    
    for step in range(10):  # Recommend up to 10 courses
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        state_key = tuple(sorted(state))
        q_values = q_table[state_key]
        
        if q_values is None:
            break
            
        # Create a mapping from action to its index in the action space
        action_to_index = {action: idx for idx, action in enumerate(env.action_space)}
        action_q = {action: q_values[action_to_index[action]] for action in valid_actions}
        best_action = max(action_q.items(), key=lambda x: x[1])[0]
        path.append(best_action)
        
        # Get course details
        course = cpn.nodes[best_action]
        credits += course.get('credits', 3)
        
        # Display course info
        print(f"\n{step+1}. {course.get('original_id', best_action)} - {course.get('name', '')}")
        print(f"   Type: {course.get('type', '').title()}")
        print(f"   Track: {course.get('track', 'None')}")
        print(f"   Credits: {course.get('credits', 3)}")
        prereqs = list(cpn.predecessors(best_action))
        print(f"   Prerequisites: {', '.join(prereqs) if prereqs else 'None'}")
        print(f"   Q-value: {action_q[best_action]:.2f}")
        
        # Update state
        state = env.step(best_action)[0]
    
    print("\n" + "-"*50)
    print(f"Total Credits: {credits}")
    print(f"Projected GPA: {student_profile['current_gpa']:.2f}")
    print("="*50 + "\n")

def main():
    # Sample student profile
    student_profile = {
        'current_gpa': 3.2,
        'interests': ['Big_Data', 'Systems'],
        'completed_courses': ['CSCI 304', 'CSCI 207'],
        'graduation_requirements': {
            'total_credits': 120,
            'core_courses': 10,
            'electives': 6
        }
    }
    
    # Student priorities
    priorities = {
        'focus': 'balanced',
        'min_gpa': 3.0,
        'preferred_tracks': ['Big_Data', 'Media_Informatics'],
        'workload_preference': 'medium'
    }
    
    print("Building Course Prerequisite Network...")
    cpn = create_real_cpn()
    
    print("\nCreating Training Environment...")
    env = CurriculumEnvironment(cpn, student_profile, priorities)
    
    print("\nTraining Q-Learning Agent (1000 episodes)...")
    agent = train_agent(env)
    
    print("\nSaving trained model...")
    agent.save_q_table("real_academic_advisor_qtable.pkl")
    
    # Display terminal recommendation
    display_terminal_recommendation(
        cpn,
        student_profile,
        priorities,
        "real_academic_advisor_qtable.pkl"
    )
    
    # Second example with different profile
    print("\nGenerating another recommendation example...")
    sample_profile = {
        'current_gpa': 3.5,
        'interests': ['Media_Informatics', 'Visualization'],
        'completed_courses': ['CSCI 304', 'CSCI 451'],
        'graduation_requirements': {
            'total_credits': 120,
            'core_courses': 10,
            'electives': 6
        }
    }
    
    sample_priorities = {
        'focus': 'interests',
        'min_gpa': 3.3,
        'preferred_tracks': ['Big_Data'],
        'workload_preference': 'medium'
    }
    
    display_terminal_recommendation(
        cpn,
        sample_profile,
        sample_priorities,
        "real_academic_advisor_qtable.pkl"
    )

if __name__ == "__main__":
    main()