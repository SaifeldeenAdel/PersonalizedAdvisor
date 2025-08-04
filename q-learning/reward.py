import numpy as np

def calculate_base_reward(course_features, student_profile):
    """
    Combine multiple reward components using your real course features
    """
    # GPA component (weighted more heavily for students with higher GPA goals)
    gpa_weight = 0.5 if student_profile['current_gpa'] >= 3.0 else 0.3
    gpa_reward = (course_features['expected_gpa'] / 4.0) * gpa_weight
    
    # Interest match component
    interest_weight = 0.3
    interest_reward = course_features['interest_match'] * interest_weight
    
    # Unlocks component - more valuable for courses that unlock many options
    unlocks_weight = 0.15
    unlocks_reward = (len(course_features['unlocks_courses']) / 10) * unlocks_weight
    
    # Workload component - adjust based on student preference
    workload_weight = 0.05
    if student_profile.get('workload_preference', 'medium') == 'low':
        workload_reward = (6 - course_features['workload']) / 5 * workload_weight * 2
    elif student_profile.get('workload_preference', 'medium') == 'high':
        workload_reward = (course_features['workload'] - 1) / 5 * workload_weight
    else:  # medium
        workload_reward = (3 - abs(course_features['workload'] - 3)) / 3 * workload_weight
    
    # Compulsory course bonus
    compulsory_bonus = 0.1 if course_features['compulsory'] else 0
    
    # Track alignment bonus
    track_bonus = 0.15 if course_features['track'] in student_profile.get('preferred_tracks', []) else 0
    
    base_reward = (
        gpa_reward +
        interest_reward +
        unlocks_reward +
        workload_reward +
        compulsory_bonus +
        track_bonus
    )
    
    return base_reward

def apply_priority_weights(base_reward, priorities):
    """
    Adjust reward based on student priorities
    """
    priority_factor = 1.0
    
    if priorities['focus'] == 'gpa':
        priority_factor *= 1.5
    elif priorities['focus'] == 'speed':
        priority_factor *= 0.8  # might take slightly harder courses to graduate faster
    elif priorities['focus'] == 'interests':
        priority_factor *= 1.2
    
    # Additional adjustments based on other priorities
    if priorities.get('min_gpa', 0) > 3.5:
        priority_factor *= 1.2
    
    return base_reward * priority_factor

def compute_reward(course_features, student_profile, priorities):
    base = calculate_base_reward(course_features, student_profile)
    return apply_priority_weights(base, priorities)