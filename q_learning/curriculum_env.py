class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities):
        self.cpn = cpn 
        self.student_profile = student_profile
        self.priorities = priorities
        self.state = None  # Would represent current transcript
        
    def reset(self):
        """Return to initial state (no courses taken)"""
        pass
        
    def step(self, action):
        """Simulate taking a set of courses"""
        pass
        
    def get_course_features(self, course_id):
        """Return features for a specific course"""
        return {
            'expected_gpa': ...,
            'interest_match': ...,
            'unlocks_courses': ...
        }