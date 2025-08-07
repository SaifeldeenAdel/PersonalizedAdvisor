import numpy as np
import pandas as pd
from reward import compute_reward


class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities, course_features_df, grade_predictor):
        """
        cpn: Course Prerequisite Network (graph structure)
        student_profile: Dictionary with student information
        priorities: Dictionary with student priorities
        course_features_df: DataFrame with course features
        grade_predictor: Function to predict grades
        """
        self.cpn = cpn
        self.student_profile = student_profile
        self.priorities = priorities
        self.course_features_df = course_features_df
        self.grade_predictor = grade_predictor
        self.state = set()  # Set of completed courses
        self.current_gpa = student_profile.get('current_gpa', 3.0)
        self.completed_courses_grades = {}  # Track grades of completed courses
        self.total_attempted = 0
        self.total_failed = 0
        
    def reset(self):
        """Return to initial state (no courses taken)"""
        self.state = set()
        self.completed_courses_grades = {}
        return self._get_state_representation()
        
    def step(self, action):
        """
        Simulate taking a course
        Returns: next_state, reward, done, info
        """
        self.total_attempted += 1
        # Predict grade for this course
        predicted_grade = self._predict_grade_for_course(action)

        if predicted_grade < 1:  
            self.total_failed += 1
        
        # Update fail rate in student profile
        self.student_profile['fail_rate'] = self.total_failed / self.total_attempted
        
        # Update state and track grade
        self.state.add(action)
        self.completed_courses_grades[action] = predicted_grade
        self.current_gpa = self._calculate_current_gpa()
        
        # Get course features including predicted grade
        course_features = self.get_course_features(action)
        course_features['predicted_grade'] = predicted_grade
        
        reward = compute_reward(course_features, self.student_profile, self.priorities, self.current_gpa)
        
        # Check if graduation requirements are met
        done = self._check_graduation()
        
        next_state = self._get_state_representation()
        
        return next_state, reward, done, {'predicted_grade': predicted_grade, 'fail_rate': self.student_profile['fail_rate']}
    
    def _predict_grade_for_course(self, course_id):
        """Predict grade for a course using the loaded model"""
        # Get course features from DataFrame
        course_data = self.course_features_df[self.course_features_df['Course ID'] == course_id].iloc[0]
        
        # Prepare input for prediction
        prediction_input = {
            'CREDIT': float(course_data['CREDIT']),
            'fail_rate': float(course_data['fail_rate']),
            'average_grade': float(course_data['average_grade']),
            'student_avg_grade': float(self.student_profile.get('current_gpa', 3.0)),
            'student_fail_rate': float(self.student_profile.get('fail_rate', 0.0)),
            'CourseCode': course_data['CourseCode']
        }
        prediction_input = pd.DataFrame([prediction_input])
        
        return self.grade_predictor.predict(prediction_input)[0]
    
    def _calculate_current_gpa(self):
        """Calculate current GPA based on completed courses"""
        if not self.completed_courses_grades:
            return self.student_profile.get('current_gpa', 3.0)
        
        total_credits = 0
        weighted_sum = 0
        
        for course, grade in self.completed_courses_grades.items():
            # Get course credits (assuming all courses are same credit for simplicity)
            # In real implementation, you'd look this up from course data
            credits = 3  # Default value
            weighted_sum += grade * credits
            total_credits += credits
            
        return weighted_sum / total_credits
    
    def get_course_features(self, course_id):
        """Return features for a specific course including predicted grade"""
        course_data = self.course_features_df[self.course_features_df['Course ID'] == course_id].iloc[0]
        
        return {
            'course_id': course_id,
            'credit': float(course_data['CREDIT']),
            'fail_rate': float(course_data['fail_rate']),
            'average_grade': float(course_data['average_grade']),
            'unlocks_courses': list(self.cpn.successors(course_id)),
            'workload': float(course_data.get('workload', 3))  # Default to medium workload if not available
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