import numpy as np

def calculate_base_reward(course_features, student_profile):
    """
    Combine multiple reward components:
    - GPA expectation (higher is better)
    - Interest match (higher is better)
    - Course unlocks (more unlocks is better)
    - Workload (lower is better)
    """
    gpa_reward = course_features['expected_gpa'] / 4.0  # normalize to [0,1]
    interest_reward = course_features['interest_match']
    unlocks_reward = 0.1 * len(course_features['unlocks_courses'])  # weight unlocks
    
    # Penalize difficult courses for students with low GPA
    gpa_adjustment = 1 + (student_profile['current_gpa'] - 2.5) / 2.5  # scales 0.5-1.5
    
    base_reward = (
        0.4 * gpa_reward * gpa_adjustment +
        0.3 * interest_reward +
        0.2 * unlocks_reward -
        0.1 * course_features['workload']  # workload is 1-5 scale
    )
    
    return base_reward

def apply_priority_weights(base_reward, priorities):
    """
    Adjust reward based on student priorities:
    - priorities is a dict with weights for different objectives
    """
    # Example priorities might be {'gpa': 0.7, 'speed': 0.2, 'interests': 0.1}
    # In this simple version, we'll just scale the base reward
    priority_factor = 1.0
    if priorities['focus'] == 'gpa':
        priority_factor *= 1.5
    elif priorities['focus'] == 'speed':
        priority_factor *= 0.8  # might take slightly harder courses to graduate faster
    
    return base_reward * priority_factor

def compute_reward(course_features, student_profile, priorities):
    base = calculate_base_reward(course_features, student_profile)
    return apply_priority_weights(base, priorities)