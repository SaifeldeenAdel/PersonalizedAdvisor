def calculate_base_reward(course_features, student_profile, current_gpa):
    """
    Combine multiple reward components with predicted grade:
    - Predicted grade (higher is better)
    - Interest match (higher is better)
    - Course unlocks (more unlocks is better)
    - Compulsory status (required courses get bonus)
    - GPA impact (how this affects overall GPA)
    """
    # Normalize predicted grade (assuming 0-4 scale)
    grade_reward = course_features['predicted_grade'] / 4.0
    
    # Calculate GPA impact (weighted by credits)
    completed_credits = student_profile.get('completed_credits', 0)
    course_credits = course_features.get('credits', 3)
    
    current_grade_points = current_gpa * completed_credits
    total_new_credits = completed_credits + course_credits
    
    # Handle division by zero case (when no credits completed and course has 0 credits)
    if total_new_credits == 0:
        gpa_impact = 0  # or course_features['predicted_grade'] / 4.0 if you prefer
    else:
        new_gpa = (current_grade_points + course_features['predicted_grade'] * course_credits) / total_new_credits
        gpa_impact = new_gpa - current_gpa
    
    interest_reward = course_features.get('interest_match', 0.5)
    unlocks_reward = min(0.5, 0.1 * len(course_features['unlocks_courses']))  # Cap at 0.5

    compulsory_bonus = 0.2 if course_features.get('is_compulsory', False) else 0

    base_reward = (
        0.4 * grade_reward +
        0.3 * gpa_impact * 4.0 +  # Scale to similar range as grade_reward
        0.2 * interest_reward +
        0.1 * unlocks_reward +
        compulsory_bonus
    )
    
    # Ensure reward stays in reasonable bounds
    return max(0, min(1, base_reward))

def apply_priority_weights(base_reward, priorities, course_features):
    """
    Adjust reward based on student priorities and course features
    """
    priority_factor = 1.0
    
    if priorities.get('focus') == 'gpa':
        # More weight on GPA impact and predicted grade
        #TODO : update based on gpa increase
        priority_factor *= 1.0 + (0.5 * (course_features['predicted_grade'] / 4.0))

    elif priorities.get('focus') == 'speed':
        # More weight on courses that unlock many others
        priority_factor *= 1.0 + (0.1 * len(course_features['unlocks_courses']))
    
    elif priorities.get('focus') == 'interests':
        # More weight on interest match
        priority_factor *= 1.0 + course_features.get('interest_match', 0)
    # Apply track preference multiplier if specified
    if 'preferred_tracks' in priorities:
        if course_features.get('track') in priorities['preferred_tracks']:
            priority_factor *= 1.2

    return base_reward * priority_factor


def compute_reward(course_features, student_profile, priorities, current_gpa):
    base = calculate_base_reward(course_features, student_profile, current_gpa)
    return apply_priority_weights(base, priorities, course_features)