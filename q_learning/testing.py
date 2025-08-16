# test.py
from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
import joblib
from cpn import build_graph_unique, load_data
from train import display_recommendation, create_cpn

def main():
    # Real student profile
    real_student = {
        'current_gpa': 3.85,
        'interests': ['Big_Data'],
        'completed_courses': ['CSCI101', 'CSCI102', 'ENGL101', 'MATH100'],
        'grades': {
            'CSCI101': 3.7,
            'CSCI102': 3.7,
            'ENGL101': 4.0,
            'MATH100': 4.0
        },
        'current_semester': 2,
        'graduation_requirements': {
            'total_credits': 135,
            'core_courses': 20,
            'electives': 4
        }
    }

    priorities = {
        'focus': 'speed',
        'min_gpa': 2.0,
        'preferred_tracks': ['Big_Data']
    }

    print("Building Course Prerequisite Network...")
    cpn = create_cpn()

    print("\nLoading Grade Prediction Model...")
    grade_predictor = joblib.load('helper/poly_reg_pipeline.pkl')

    print("\nCreating Environment...")
    env = CurriculumEnvironment(
        cpn=cpn,
        student_profile=real_student,
        priorities=priorities,
        grade_predictor=grade_predictor
    )

    print("\nLoading Trained Q-Learning Agent...")
    agent = QLearningAgent()
    agent.load_q_table("academic_advisor_qtable_final.pkl")

    print("\nGenerating Personalized Recommendation...")
    display_recommendation(agent, env)

if __name__ == "__main__":
    main()
