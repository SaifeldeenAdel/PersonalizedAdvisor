import numpy as np
import pandas as pd
from reward import compute_reward


class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities, grade_predictor):
        """
        cpn: Course Prerequisite Network (graph structure)
        student_profile: Dictionary with student information
        priorities: Dictionary with student priorities
        grade_predictor: Function to predict grades
        """
        self.cpn = cpn
        self.student_profile = student_profile
        self.priorities = priorities
        self.grade_predictor = grade_predictor
        self.state = set()  # Set of completed courses
        self.current_gpa = student_profile.get('current_gpa', 3.0)
        self.completed_courses_grades = {}  # Track grades of completed courses
        self.credit_count = 0
        self.total_attempted = 0
        self.total_failed = 0
        self.action_space = list(cpn.nodes()) 
        
    def reset(self):
        """Return to initial state (no courses taken)"""
        self.state = set()
        self.completed_courses_grades = {}
        self.credit_count = 0
        self.total_attempted = 0
        self.total_failed = 0
        return self._get_state_representation()
        
    def step(self, action):
        """
        Simulate taking a course
        Returns: next_state, reward, done, info
        """
        self.total_attempted += 1

        # Get course data from CPN node
        course_data = self.cpn.nodes[action]
        course_credits = course_data.get('credits', 3)
        self.credit_count += course_credits
        
        # Predict grade for this course
        predicted_grade = self._predict_grade_for_course(action, course_data)

        if predicted_grade < 1:  
            self.total_failed += 1
        
        # Update fail rate in student profile
        self.student_profile['fail_rate'] = self.total_failed / max(1, self.total_attempted)
        
        # Update state and track grade
        self.state.add(action)
        self.completed_courses_grades[action] = predicted_grade
        self.current_gpa = self._calculate_current_gpa()
        
        # Get course features including predicted grade
        course_features = self.get_course_features(action, course_data)
        course_features['predicted_grade'] = predicted_grade
        
        # Compute reward
        reward = compute_reward(course_features, self.student_profile, self.priorities, self.current_gpa)
        
        # Check if graduation requirements are met
        done = self._check_graduation()
        
        next_state = self._get_state_representation()
        
        return next_state, reward, done, {
            'predicted_grade': predicted_grade,
            'fail_rate': self.student_profile['fail_rate'],
            'credits': self.credit_count,
            'gpa': self.current_gpa
        }
    
    def _predict_grade_for_course(self, course_id, course_data):
        """Predict grade for a course using the loaded model"""
        # Prepare input for prediction from CPN node data
        prediction_input = {
            'CREDIT': float(course_data.get('credits', 3)),
            'fail_rate': float(course_data.get('fail_rate', 0.1)),
            'average_grade': float(course_data.get('average_grade', 3.0)),
            'student_avg_grade': float(self.student_profile.get('current_gpa', 3.0)),
            'student_fail_rate': float(self.student_profile.get('fail_rate', 0.0)),
            'CourseCode': course_data.get('CourseCode')
        }
        prediction_input = pd.DataFrame([prediction_input])
        
        prediction = self.grade_predictor.predict(prediction_input)[0]
        return max(0, min(4, prediction))  # Assuming 0-4 grading scale
        
    def _calculate_current_gpa(self):
        """Calculate current GPA based on completed courses"""
        if not self.completed_courses_grades:
            return self.student_profile.get('current_gpa', 3.0)
        
        total_credits = 0
        weighted_sum = 0
        
        for course, grade in self.completed_courses_grades.items():
            credits = self.cpn.nodes[course].get('credits', 3)
            weighted_sum += grade * credits
            total_credits += credits
            
        return weighted_sum / total_credits if total_credits > 0 else self.student_profile.get('current_gpa', 3.0)
    
    
    def get_course_features(self, course_id, course_data=None):
            """Return features for a specific course"""
            if course_data is None:
                course_data = self.cpn.nodes[course_id]
            
            # Calculate interest match based on student interests and course category/track
            interest_match = 0
            if 'interests' in self.student_profile:
                student_interests = self.student_profile['interests']
                course_category = course_data.get('category', '').lower()
                course_track = course_data.get('track', '').lower()
                
                for interest in student_interests:
                    if interest.lower() in course_category or interest.lower() in course_track:
                        interest_match += 0.5  # Increase for each match
            
            # Normalize interest match to 0-1 range
            interest_match = min(1.0, interest_match)
            
            features = {
                'course_id': course_id,
                'credits': course_data.get('credits', 3),
                'fail_rate': course_data.get('fail_rate', 0.1),  # Default 10% fail rate
                'average_grade': course_data.get('average_grade', 3.0),  # Default average grade
                'unlocks_courses': list(self.cpn.successors(course_id)),
                'course_type': course_data.get('course_type', 'elective'),
                'track': course_data.get('track', 'None'),
                'is_compulsory': course_data.get('is_compulsory', False),
                'interest_match': interest_match,
                'course_name': course_data.get('course_name'),
                'CourseCode': course_data.get('CourseCode')
            }
            return features

    
    def _get_state_representation(self):
        """Convert state to a format usable by the agent"""
        return frozenset(self.state)
    
    def _check_graduation(self):
        """Check if all degree requirements are met"""
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
            
        # Check minimum GPA requirement if specified
        if 'min_gpa' in self.priorities and self.current_gpa < self.priorities['min_gpa']:
            return False
            
        return True
    
    def get_valid_actions(self):
        """Return courses that can be taken next based on prerequisites and track requirements"""
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