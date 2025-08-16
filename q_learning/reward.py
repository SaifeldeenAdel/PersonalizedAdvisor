# reward.py

def calculate_base_reward(course_features, student_profile, current_gpa, current_semester):
    """
    Calculates the reward for taking a single course, with a strong emphasis on out_degree.
    """
    # Grade reward: Higher predicted grade is better
    grade_reward = course_features['predicted_grade'] / 4.0

    # NEW: Heavily reward courses that unlock future options.
    # The out_degree is now a primary driver of the reward.
    out_degree_reward = course_features.get('out_degree', 0)

    # Interest reward: 1 if it matches, 0 otherwise
    interest_reward = course_features.get('interest_match', 0)
    
    # Level appropriateness penalty
    course_level = course_features.get('course_level', 1)
    ideal_level = (current_semester // 2) + 1
    level_diff = abs(course_level - ideal_level)
    level_penalty = -0.6 * (level_diff ** 2)

    # NEW WEIGHTS: Give more importance to out_degree
    base_reward = (
        0.1 * grade_reward +
        0.3 * out_degree_reward +  #<-- Increased from 0.3 to 0.4
        0.1 * interest_reward +
        level_penalty
    )
    
    return base_reward

def apply_priority_weights(base_reward, priorities, course_features):
    # """Adjust reward based on student priorities."""
    # priority_factor = 1.0
    
    # if priorities.get('focus') == 'gpa':
    #     priority_factor += 0.5 * (course_features['predicted_grade'] / 4.0)
    # elif priorities.get('focus') == 'speed':
    #     # Speed focus now directly boosts the out_degree reward's effect
    #     priority_factor += 0.5 * (0.1 * course_features.get('out_degree', 0))
    # elif priorities.get('focus') == 'interests':
    #     priority_factor += 0.5 * course_features.get('interest_match', 0)

    # if 'preferred_tracks' in priorities and course_features.get('track') in priorities['preferred_tracks']:
    #     priority_factor *= 1.2

    # return base_reward * priority_factor
    return base_reward

def compute_reward(course_features, student_profile, priorities, current_gpa, semester):
    base = calculate_base_reward(course_features, student_profile, current_gpa, semester)
    reward = apply_priority_weights(base, priorities, course_features)
    return reward