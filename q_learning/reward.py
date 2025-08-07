def calculate_base_reward(course_features, student_profile, current_gpa):
    """
    Combine multiple reward components with predicted grade:
    - Predicted grade (higher is better)
    - Interest match (higher is better)
    - Course unlocks (more unlocks is better)
    - Workload (lower is better)
    - GPA impact (how this affects overall GPA)
    """
    # Normalize predicted grade (assuming 0-4 scale)
    grade_reward = course_features['predicted_grade'] / 4.0
    
    # Calculate GPA impact (how this course would affect current GPA)
    current_total = current_gpa * len(student_profile.get('completed_courses', []))
    new_gpa = (current_total + course_features['predicted_grade']) / (len(student_profile.get('completed_courses', [])) + 1)
    gpa_impact = new_gpa - current_gpa
    
    interest_reward = course_features.get('interest_match', 0.5)  # Default if not available
    unlocks_reward = 0.1 * len(course_features['unlocks_courses'])
    
    base_reward = (
        0.5 * grade_reward +
        0.2 * gpa_impact * 4.0 +  # Scale to similar range as other components
        0.2 * interest_reward +
        0.1 * unlocks_reward -
        0.1 * course_features.get('workload', 3) / 5.0  # Normalize workload to 0-1
    )
    
    return base_reward

def apply_priority_weights(base_reward, priorities, predicted_grade):
    """
    Adjust reward based on student priorities and predicted grade
    """
    priority_factor = 1.0
    
    if priorities['focus'] == 'gpa':
        # More weight on courses that improve GPA
        priority_factor *= 1.0 + (predicted_grade - 2.5) / 5.0  # Scale adjustment
    
    elif priorities['focus'] == 'speed':
        # More weight on courses that unlock many others
        priority_factor *= 1.2
    
    elif priorities['focus'] == 'interests':
        # More weight on interest match
        priority_factor *= 1.3
    
    return base_reward * priority_factor

def compute_reward(course_features, student_profile, priorities, current_gpa):
    base = calculate_base_reward(course_features, student_profile, current_gpa)
    return apply_priority_weights(base, priorities, course_features['predicted_grade'])