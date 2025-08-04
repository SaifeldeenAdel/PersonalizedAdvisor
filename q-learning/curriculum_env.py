import numpy as np
from reward import compute_reward


class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities):
        self.cpn = cpn
        self.student_profile = student_profile
        self.priorities = priorities
        self.state = set()
        self.credit_count = 0
        self.action_space = list(cpn.nodes())  # Add this line
    def reset(self):
        """Return to initial state (no courses taken)"""
        self.state = set()
        self.credit_count = 0
        return self._get_state_representation()
        
    def step(self, action):
        """
        Simulate taking a course
        Returns: next_state, reward, done, info
        """
        self.state.add(action)
        course_credits = self.cpn.nodes[action].get('credits', 3)  # Default to 3 if not specified
        self.credit_count += course_credits
        
        course_features = self.get_course_features(action)
        reward = compute_reward(course_features, self.student_profile, self.priorities)
        
        # Check if graduation requirements are met
        done = self._check_graduation()
        
        next_state = self._get_state_representation()
        
        return next_state, reward, done, {'credits': self.credit_count}
    
    def get_course_features(self, course_id):
        """Return features for a specific course from your real data"""
        node_data = self.cpn.nodes[course_id]
        
        # Calculate interest match based on student interests and course category/track
        interest_match = 0
        if 'interests' in self.student_profile:
            student_interests = self.student_profile['interests']
            course_category = node_data.get('category', '').lower()
            course_track = node_data.get('track', '').lower()
            
            for interest in student_interests:
                if interest.lower() in course_category or interest.lower() in course_track:
                    interest_match += 0.5  # Increase for each match
        
        # Normalize interest match to 0-1 range
        interest_match = min(1.0, interest_match)
        
        return {
            'expected_gpa': self._estimate_expected_gpa(course_id),
            'interest_match': interest_match,
            'unlocks_courses': list(self.cpn.successors(course_id)),
            'workload': node_data.get('workload', 3),  # Default to medium workload
            'credits': node_data.get('credits', 3),
            'type': node_data.get('type', 'elective'),
            'track': node_data.get('track', 'None'),
            'compulsory': node_data.get('compulsory', False)
        }
    
    def _estimate_expected_gpa(self, course_id):
        """Estimate expected GPA based on student's current GPA and course difficulty"""
        # This should be enhanced with historical data if available
        base_gpa = self.student_profile['current_gpa']
        course_level = self.cpn.nodes[course_id].get('level', 1)
        
        # Simple estimation: higher level courses are slightly harder
        level_adjustment = 0.95 ** (course_level - 1)
        return max(1.0, min(4.0, base_gpa * level_adjustment))
    
    def _get_state_representation(self):
        """Convert state to a format usable by the agent"""
        return frozenset(self.state)
    
    def _check_graduation(self):
        """Check if all degree requirements are met using your real requirements"""
        # Check credit requirement
        if self.credit_count < self.student_profile['graduation_requirements']['total_credits']:
            return False
            
        # Check core courses
        core_courses_taken = sum(
            1 for course in self.state 
            if self.cpn.nodes[course].get('type') == 'core'
        )
        if core_courses_taken < self.student_profile['graduation_requirements']['core_courses']:
            return False
            
        # Check elective courses
        elective_courses_taken = sum(
            1 for course in self.state 
            if self.cpn.nodes[course].get('type') == 'elective'
        )
        if elective_courses_taken < self.student_profile['graduation_requirements']['electives']:
            return False
            
        return True
    
    def get_valid_actions(self):
        """Return courses that can be taken next based on prerequisites"""
        valid_actions = []
        for course in self.cpn.nodes():
            # Check if all prerequisites are met
            prereqs = list(self.cpn.predecessors(course))
            if (all(p in self.state for p in prereqs) and 
                course not in self.state and
                self._meets_track_requirements(course)):
                valid_actions.append(course)
        return valid_actions
    
    def _meets_track_requirements(self, course_id):
        """Check if course matches student's track preferences"""
        if not self.priorities.get('preferred_tracks'):
            return True
            
        course_track = self.cpn.nodes[course_id].get('track', 'None')
        return (course_track in self.priorities['preferred_tracks'] or 
                course_track == 'None' or
                self.cpn.nodes[course_id].get('compulsory', False))