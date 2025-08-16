# train.py
from agent import QLearningAgent
from curriculum_env import CurriculumEnvironment
import joblib
from cpn import build_graph_unique, load_data
from reward import compute_reward

def get_semester_template(semester_num):
    base_template = {
        'Computer Science': (1, 3),  # Min 1, Max 4 CS courses
        'Mathematics': (1, 3),
        'Natural Sciences': (0, 1),
        'Humanities': (0, 1),
        'Physics': (0, 1),
        'English language': (0, 1),
        'Electrical&Computer Engineering': (0, 1),
        'Social Sciences': (0, 1),
    }

    if semester_num == 7:
        return {**base_template, 'Senior Project I': (1, 1)}
    elif semester_num == 8:
        return {**base_template, 'Senior Project II': (1, 1)}
    else:
        return base_template

def create_cpn():
    catalog = load_data()
    return build_graph_unique(catalog)

def categorize_courses(available_courses, env):
    subject_categories = {
        'Computer Science': [],
        'Mathematics': [],
        'Natural Sciences': [],
        'Humanities': [],
        'Physics': [],
        'English language': [],
        'Electrical&Computer Engineering': [],
        'Social Sciences': [],
        'Senior Project I': [],
        'Senior Project II': []
    }

    type_categories = {
        'core': [],
        'elective': [],
        'general': []
    }

    for course_id in available_courses:
        node_data = env.cpn.nodes[course_id]

        # Senior Project mapping by course ID
        if course_id == "CSCI495":
            subject_categories['Senior Project I'].append(course_id)
        elif course_id == "CSCI496":
            subject_categories['Senior Project II'].append(course_id)
        else:
            subject = node_data.get('category', 'general')
            if subject in subject_categories:
                subject_categories[subject].append(course_id)
            else:
                subject_categories['general'].append(course_id)

        # Normalize course type
        course_type = node_data.get('type', 'elective').lower()
        if course_type in type_categories:
            type_categories[course_type].append(course_id)
        else:
            type_categories['elective'].append(course_id)

    return {'by_subject': subject_categories, 'by_type': type_categories}

def filter_by_semester_template(available_courses, env):
    semester_template = get_semester_template(env.semester)

    # If Senior Projects aren’t allowed yet, remove them
    if env.semester < 7:
        return [c for c in available_courses if c not in ("CSCI495", "CSCI496")]
    return available_courses

def train_agent(env, episodes=1500, verbose=True):
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)

    for episode in range(episodes):
        base_state = env.reset()
        total_reward = 0
        done = False

        while not done:
            semester_plan = []
            available_courses = filter_by_semester_template(env.get_valid_actions(), env)

            for _ in range(6):
                if not available_courses:
                    break

                current_state = (base_state, tuple(sorted(semester_plan)))
                chosen_course = agent.choose_action(current_state, available_courses)

                if not chosen_course:
                    continue

                # Immediate reward
                course_features = env.get_course_features(chosen_course)
                course_features['predicted_grade'] = env._predict_grade_for_course(
                    chosen_course, env.cpn.nodes[chosen_course]
                )
                immediate_reward = compute_reward(
                    course_features, env.student_profile, env.priorities, env.current_gpa, env.semester
                )
                total_reward += immediate_reward

                next_state = (base_state, tuple(sorted(semester_plan + [chosen_course])))

                agent.update(
                    state=current_state,
                    action=chosen_course,
                    reward=immediate_reward,
                    next_state=next_state,
                    next_valid_actions=[c for c in available_courses if c != chosen_course]
                )

                semester_plan.append(chosen_course)
                available_courses.remove(chosen_course)

            if not semester_plan:
                done = True
                total_reward -= 50
                continue

            next_base_state, semester_end_reward, done, _ = env.step(semester_plan)
            total_reward += semester_end_reward

            if semester_end_reward != 0:
                final_state = (base_state, tuple(sorted(semester_plan[:-1])))
                final_action = semester_plan[-1]
                agent.update(
                    final_state, final_action, semester_end_reward, (next_base_state, ()), env.get_valid_actions()
                )

            base_state = next_base_state

        if verbose:
            print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.2f}, "
                  f"Credits: {env.credit_count}, GPA: {env.current_gpa:.2f}")

    return agent

def display_recommendation(agent, env):
    agent.epsilon = 0
    base_state = env.reset()
    done = False

    print("\n===== Student Profile =====")
    for k, v in env.student_profile.items():
        print(f"{k}: {v}")
    print("===========================\n")

    while not done and env.semester <= env.max_semesters:
        print(f"----- Semester {env.semester} -----")
        semester_plan = []
        available_courses = filter_by_semester_template(env.get_valid_actions(), env)

        categorized = categorize_courses(available_courses, env)
        subject_categories = categorized['by_subject']
        semester_template = get_semester_template(env.semester)

        # Pick required courses first
        for category, (min_req, _) in semester_template.items():
            candidates = subject_categories.get(category, [])
            for _ in range(min_req):
                if not candidates:
                    break
                current_state = (base_state, tuple(sorted(semester_plan)))
                chosen = agent.choose_action(current_state, candidates)
                if chosen:
                    semester_plan.append(chosen)
                    candidates.remove(chosen)
                    if chosen in available_courses:
                        available_courses.remove(chosen)

        # Fill remaining slots
        while len(semester_plan) < 5 and available_courses:
            current_counts = {}
            for course in semester_plan:
                cat = env.cpn.nodes[course].get('category', 'general')
                current_counts[cat] = current_counts.get(cat, 0) + 1

            available_cats = []
            for category, (min_req, max_req) in semester_template.items():
                current = current_counts.get(category, 0)
                cat_courses = [c for c in subject_categories.get(category, []) if c not in semester_plan]
                if current < max_req and cat_courses:
                    available_cats.append(category)

            if not available_cats:
                break

            category = 'Computer Science' if 'Computer Science' in available_cats \
                       else 'Mathematics' if 'Mathematics' in available_cats \
                       else available_cats[0]

            candidates = [c for c in subject_categories.get(category, []) if c not in semester_plan]
            if candidates:
                current_state = (base_state, tuple(sorted(semester_plan)))
                chosen = agent.choose_action(current_state, candidates)
                if chosen:
                    semester_plan.append(chosen)
                    available_courses.remove(chosen)
            else:
                continue

        # Optional 6th course
        if len(semester_plan) == 5 and available_courses:
            remaining = [c for c in available_courses if c not in semester_plan]
            if remaining:
                current_state = (base_state, tuple(sorted(semester_plan)))
                chosen = agent.choose_action(current_state, remaining)
                if chosen:
                    semester_plan.append(chosen)

        if not semester_plan:
            print("No valid courses could be recommended. The student may not be able to graduate.")
            break

        credits = sum(env.cpn.nodes[c].get('credits', 3) for c in semester_plan)
        print(f"Recommended Courses: {semester_plan}")
        print(f"Semester Credits: {credits}")

        next_state, _, done, info = env.step(semester_plan)
        base_state = next_state

        print(f"End of Semester GPA: {info['gpa']:.2f} | Total Credits: {info['credits']}")
        print("-" * 25)

    if env._check_graduation():
        print("\n🎉 Graduation Requirements Met!")
    else:
        print("\nMax semesters reached. Graduation requirements not fully met.")

    print(f"Final GPA: {env.current_gpa:.2f}")
    print(f"Total Credits: {env.credit_count}")

def main():
    student_profile = {
        'current_gpa': 3.5,
        'interests': ['Media_Informatics', 'Visualization'],
        'completed_courses': [],
        'current_semester': 1,
        'graduation_requirements': {'total_credits': 135, 'core_courses': 20, 'electives': 4}
    }
    priorities = {'focus': 'speed', 'min_gpa': 2.0, 'preferred_tracks': ['Big_Data']}

    print("Building Course Prerequisite Network...")
    cpn = create_cpn()
    print("\nLoading Grade Prediction Model...")
    grade_predictor = joblib.load('helper/poly_reg_pipeline.pkl')
    print("\nCreating Environment...")
    env = CurriculumEnvironment(cpn=cpn, student_profile=student_profile, priorities=priorities,
                                grade_predictor=grade_predictor)

    print("\nTraining Q-Learning Agent...")
    agent = train_agent(env, episodes=2000)

    output_file = "academic_advisor_qtable_final.pkl"
    agent.save_q_table(output_file)
    display_recommendation(agent, env)

if __name__ == "__main__":
    main()
