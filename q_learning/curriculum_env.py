import itertools
import random
import pandas as pd
from reward import compute_reward

class CurriculumEnvironment:
    def __init__(self, cpn, student_profile, priorities, grade_predictor, max_semesters=8):
        self.cpn = cpn
        self.student_profile_initial = student_profile.copy()
        self.priorities = priorities
        self.grade_predictor = grade_predictor
        self.max_semesters = max_semesters
        self.reset()

    def reset(self):
        self.student_profile = self.student_profile_initial.copy()
        self.state = set(self.student_profile.get('completed_courses', []))
        self.completed_courses_grades = {} # Store grades for GPA calculation
        
        # Calculate initial credit count and GPA from pre-completed courses
        initial_credits = 0
        weighted_sum = 0
        for course in self.state:
            credits = self.cpn.nodes[course].get('credits', 3)
            grade = self.student_profile.get('grades', {}).get(course, 3.0) # Assume B if not specified
            initial_credits += credits
            weighted_sum += grade * credits
            self.completed_courses_grades[course] = grade

        self.credit_count = initial_credits
        self.current_gpa = self.student_profile.get('current_gpa', 3.0)
        
        self.total_attempted = len(self.state)
        self.total_failed = 0
        self.semester = int(self.student_profile.get('current_semester', 1))
        
        return self._get_state_representation()

    def step(self, action_set):
        """
        Processes an entire semester's worth of courses (action_set).
        action_set: A list/tuple of course IDs for the semester.
        """
        semester_reward = 0
        semester_credits = 0

        for course_id in action_set:
            self.total_attempted += 1
            course_data = self.cpn.nodes[course_id]
            course_credits = course_data.get('credits', 3)
            
            # Predict grade and update state
            predicted_grade = self._predict_grade_for_course(course_id, course_data)
            self.state.add(course_id)
            self.completed_courses_grades[course_id] = predicted_grade
            
            if predicted_grade < 1.0: # Assuming 1.0 is the passing grade
                self.total_failed += 1

            # Get features for reward calculation
            course_features = self.get_course_features(course_id, course_data)
            course_features['predicted_grade'] = predicted_grade
            
            # Accumulate reward for each course in the semester
            semester_reward += compute_reward(course_features, self.student_profile, self.priorities, self.current_gpa, self.semester)
            semester_credits += course_credits

        # Update cumulative stats after the semester
        self.credit_count += semester_credits
        self.student_profile['fail_rate'] = self.total_failed / max(1, self.total_attempted)
        self.current_gpa = self._calculate_current_gpa()

        done = False
        # Apply graduation/failure rewards
        if self._check_graduation():
            semester_reward += 200 - (self.semester - 1) * 10
            done = True
        elif self.semester > self.max_semesters:
            semester_reward -= 300   # harsher penalty than success reward
            done = True

        self.semester += 1
        next_state = self._get_state_representation()

        return next_state, semester_reward, done, {
            'gpa': self.current_gpa,
            'semester': self.semester,
            'credits': self.credit_count
        }

    def get_valid_actions(self):
        """
        Returns a list of SINGLE valid courses that can be taken.
        """
        open_courses = [
            course for course in self.cpn.nodes()
            if course not in self.state and all(p in self.state for p in self.cpn.predecessors(course))
        ]
        if self.semester < 5:
            open_courses = [c for c in open_courses if c not in ['CSCI490', 'CSCI459', 'COMM401']]
        if self.semester < 7:
            open_courses = [c for c in open_courses if 'CSCI495' not in c and 'CSCI496' not in c]
        elif self.semester == 7:
            open_courses = [c for c in open_courses if 'CSCI496' not in c]  # allow only I
        electives_taken = sum(1 for course in self.state if self.cpn.nodes[course].get('type') == 'elective')
        elective_quota = self.student_profile['graduation_requirements'].get('electives', 0)

        if electives_taken >= elective_quota:
            open_courses = [c for c in open_courses if self.cpn.nodes[c].get('type') != 'elective']

        return open_courses
    
    def _get_state_representation(self):
        """
        Enriched state representation: (completed_courses, current_semester, binned_gpa)
        """
        completed_courses_tuple = tuple(sorted(list(self.state)))
        # Bin GPA to the nearest 0.5 to keep state space manageable
        gpa_bin = round(self.current_gpa * 2) / 2
        return (completed_courses_tuple, self.semester, gpa_bin)

    def _check_graduation(self):
        reqs = self.student_profile['graduation_requirements']
        if self.credit_count < reqs['total_credits']:
            return False
        
        core_courses_taken = sum(1 for c in self.state if self.cpn.nodes[c].get('type') == 'core')
        if core_courses_taken < reqs['core_courses']:
            return False
        
        elective_courses_taken = sum(1 for c in self.state if self.cpn.nodes[c].get('type') == 'elective')
        if elective_courses_taken < reqs['electives']:
            return False
        
        if 'min_gpa' in self.priorities and self.current_gpa < self.priorities['min_gpa']:
            return False
            
        return True

    # --- Helper methods (mostly unchanged) ---
    def _predict_grade_for_course(self, course_id, course_data):
        prediction_input = {
            'CREDIT': float(course_data.get('credits', 3)),
            'fail_rate': float(course_data.get('fail_rate', 0.1)),
            'average_grade': float(course_data.get('average_grade', 3.0)),
            'student_avg_grade': float(self.current_gpa),
            'student_fail_rate': float(self.student_profile.get('fail_rate', 0.0)),
            'CourseCode': course_data.get('CourseCode')
        }
        prediction_input = pd.DataFrame([prediction_input])
        prediction = self.grade_predictor.predict(prediction_input)[0]
        return max(0, min(4, prediction))

    def _calculate_current_gpa(self):
        if not self.completed_courses_grades:
            return self.student_profile.get('current_gpa', 3.0)
        
        total_credits = 0
        weighted_sum = 0
        for course, grade in self.completed_courses_grades.items():
            credits = self.cpn.nodes[course].get('credits', 3)
            weighted_sum += grade * credits
            total_credits += credits
        return weighted_sum / total_credits if total_credits > 0 else 3.0

    def get_course_features(self, course_id, course_data=None):
        if course_data is None:
            course_data = self.cpn.nodes[course_id]
        
        interest_match = 0
        if 'interests' in self.student_profile:
            student_interests = [i.lower() for i in self.student_profile['interests']]
            course_category = course_data.get('category', '').lower()
            course_track = course_data.get('track', '').lower()
            if course_category in student_interests or course_track in student_interests:
                interest_match = 1.0

        return {
            'course_id': course_id,
            'course_level': course_data.get('course_level'),
            'credits': course_data.get('credits', 3),
            'unlocks_courses': list(self.cpn.successors(course_id)),
            'is_compulsory': course_data.get('is_compulsory', False),
            'interest_match': interest_match,
            'track': course_data.get('track', 'None'),
            'out_degree': self.cpn.out_degree(course_id),
            'category': course_data.get('category', 'GENERAL')
        }