from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
import networkx as nx
import pandas as pd
import joblib
import pickle
from collections import defaultdict
from cpn import build_graph, load_data

def create_cpn():
    """Create CPN (Curriculum Prerequisite Network)"""
    catalog = load_data()
    return build_graph(catalog)

def train_agent(env, episodes=200, verbose=True):
    """Train the Q-learning agent with grade prediction"""
    action_space = list(env.cpn.nodes())
    
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
            
            # Print progress
            if verbose and (done or len(env.state) % 5 == 0):
                print(f"Course: {action}, Predicted Grade: {info.get('predicted_grade', 'N/A'):.2f}, Reward: {reward:.2f}")
            
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Current GPA: {env.current_gpa:.2f}")
    
    return agent

def display_recommendation(cpn, student_profile, priorities, q_table_path):
    """Display prediction-based recommendation path in terminal"""
    # Load Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    q_table = defaultdict(lambda: None, q_table)
    
    # Load grade predictor
    grade_predictor = joblib.load('C:\\Users\\ahmed\\Documents\\summer25\\nile\\PersonalizedAdvisor\\helper\\poly_reg_pipeline.pkl')

    
    # Create environment
    env = CurriculumEnvironment(
        cpn=cpn,
        student_profile=student_profile,
        priorities=priorities,
        grade_predictor=grade_predictor
    )
    
    state = env.reset()
    path = []
    credits = 0
    cumulative_gpa = student_profile['current_gpa']
    completed_credits = sum(cpn.nodes[course].get('credits', 3) 
                          for course in student_profile['completed_courses'] 
                          if course in cpn.nodes)
    
    print("\n" + "="*50)
    print("PREDICTION-BASED ACADEMIC RECOMMENDATION".center(50))
    print("="*50)
    
    print("\nStudent Profile:")
    print(f"- Current GPA: {student_profile['current_gpa']:.2f}")
    print(f"- Interests: {', '.join(student_profile['interests'])}")
    print(f"- Completed Courses: {len(student_profile['completed_courses'])}")
    print(f"- Completed Credits: {completed_credits}")
    
    print("\nPriorities:")
    print(f"- Focus: {priorities['focus']}")
    print(f"- Minimum GPA: {priorities['min_gpa']}")
    
    print("\n" + "-"*50)
    print("RECOMMENDED COURSE SEQUENCE WITH GRADE PREDICTIONS".center(50))
    print("-"*50)
    
    for step in range(10):  # Recommend up to 10 courses
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        state_key = tuple(sorted(state))
        q_values = q_table[state_key]
        
        if q_values is None:
            break
            
        # Get best action based on Q-values
        action_to_index = {action: idx for idx, action in enumerate(env.action_space)}
        action_q = {action: q_values[action_to_index[action]] for action in valid_actions}
        best_action = max(action_q.items(), key=lambda x: x[1])[0]
        path.append(best_action)
        
        # Get course details and make prediction
        course = cpn.nodes[best_action]
        credits += course.get('credits', 3)
        _, _, _, info = env.step(best_action)
        predicted_grade = info.get('predicted_grade', 0)
        
        # Display detailed course info
        print(f"\n{step+1}. {course.get('original_id', best_action)} - {course.get('name', '')}")
        print(f"   Type: {course.get('type', '').title()}")
        print(f"   Track: {course.get('track', 'None')}")
        print(f"   Credits: {course.get('credits', 3)}")
        prereqs = list(cpn.predecessors(best_action))
        print(f"   Prerequisites: {', '.join(prereqs) if prereqs else 'None'}")
        print(f"   Predicted Grade: {predicted_grade:.2f}")
        print(f"   Q-value: {action_q[best_action]:.2f}")
        
        # Update state and track GPA
        state = env.state
        cumulative_gpa = env.current_gpa
    
    print("\n" + "-"*50)
    print(f"Total Recommended Credits: {credits}")
    print(f"Projected Cumulative GPA: {cumulative_gpa:.2f}")
    print("="*50 + "\n")

def main():
    # Sample student profile
    student_profile = {
        'current_gpa': 3.5,
        'interests': ['Media_Informatics', 'Visualization'],
        'completed_courses': ['CSCI304', 'CSCI451'],
        'graduation_requirements': {
            'total_credits': 120,
            'core_courses': 10,
            'electives': 6
        }
    }
    
    # Student priorities
    priorities = {
            'focus': 'interests',
            'min_gpa': 3.3,
            'preferred_tracks': ['Big_Data'],
            'workload_preference': 'medium'
        }
    
    print("Building Course Prerequisite Network...")
    cpn = create_cpn()
    
    print("\nLoading Grade Prediction Model...")
    grade_predictor = joblib.load('C:\\Users\\ahmed\\Documents\\summer25\\nile\\PersonalizedAdvisor\\helper\\poly_reg_pipeline.pkl')

    print("\nCreating Prediction-Based Environment...")
    env = CurriculumEnvironment(
        cpn=cpn,
        student_profile=student_profile,
        priorities=priorities,
        grade_predictor=grade_predictor
    )
    
    print("\nTraining Q-Learning Agent with Grade Predictions (1000 episodes)...")
    agent = train_agent(env)
    
    output_file = "academic_advisor_qtable_final.pkl"
    print(f"\nSaving trained model to {output_file}...")
    agent.save_q_table(output_file)
    
    # Display prediction-based recommendation
    display_recommendation(
        cpn,
        student_profile,
        priorities,
        output_file
    )

if __name__ == "__main__":
    main()