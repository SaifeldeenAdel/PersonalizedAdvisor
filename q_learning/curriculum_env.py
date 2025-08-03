import numpy as np
from reward import compute_reward

class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities):
        """
        cpn: Course Prerequisite Network (graph structure)
        student_profile: Dictionary with student information
        priorities: Dictionary with student priorities
        """
        self.cpn = cpn
        self.student_profile = student_profile
        self.priorities = priorities
        self.state = set()  # Set of completed courses
        
    def reset(self):
        """Return to initial state (no courses taken)"""
        self.state = set()
        return self._get_state_representation()
        
    def step(self, action):
        """
        Simulate taking a course
        Returns: next_state, reward, done, info
        """
        self.state.add(action)
        course_features = self.get_course_features(action)
        reward = compute_reward(course_features, self.student_profile, self.priorities)
        
        # Check if graduation requirements are met
        done = self._check_graduation()
        
        next_state = self._get_state_representation()
        
        return next_state, reward, done, {}
    
    def get_course_features(self, course_id):
        """Return features for a specific course"""
        # In a real implementation, this should come from our data
        return {
            'expected_gpa': np.random.normal(3.0, 0.5),  # simulated data
            'interest_match': np.random.uniform(0, 1),   # simulated
            'unlocks_courses': list(self.cpn.successors(course_id)),
            'workload': np.random.randint(1, 6)          # 1-5 scale
        }
    
    def _get_state_representation(self):
        """Convert state to a format usable by the agent"""
        return frozenset(self.state)
    
    def _check_graduation(self):
        """Check if all degree requirements are met"""
        # Simplified: assume need 30 courses to graduate, we need to add more checks, like GPA and prerequisites
        # In a real scenario, this would check if all required courses are completed
        return len(self.state) >= 30
    
    def get_valid_actions(self):
        """Return courses that can be taken next based on prerequisites"""
        valid_actions = []
        for course in self.cpn.nodes():
            # Check if all prerequisites are met
            prereqs = list(self.cpn.predecessors(course))
            if all(p in self.state for p in prereqs) and course not in self.state:
                valid_actions.append(course)
        return valid_actions