# Personalized Course Planning with Q-Learning

A research-driven project exploring how Q-Learning can assist in personalized academic advising using curriculum graphs, customized reward functions, and student-specific priorities.

---

## 📌 Overview

This project builds an intelligent advising system that learns optimal course sequences tailored to individual student priorities; maximizing GPA, graduating early, or focusing on interest areas. The core algorithm uses **Q-Learning** on a **Course Prerequisite Network (CPN)**.

---

## Methodology Progress

-   [ ] Step 1: Perform EDA and extract course-level features
-   [ ] Step 2: Build Course Prerequisite Network (CPN) and extract graph features
-   [ ] Step 3: Design reward function based on course metrics and student interests
-   [ ] Step 4: Implement priority-based reward customization
-   [ ] Step 5: Set up and implement Q-learning framework
-   [ ] Step 6: Train the model and generate recommendations

---

## Methodology (Planned Steps)

### Step 1: Exploratory Data Analysis (EDA)

-   Compute course failure rates
-   Analyze correlations with GPA (department, professor, major alignment)
-   Script: `eda.py`

### Step 2: CPN Construction (Curriculum Graph)

-   Build a directed graph of course prerequisites
-   Extract structural features:
    -   **Out-degree**: How many future courses this course unlocks
    -   **Level**: Depth in the graph
    -   **Centrality**: Influence in the curriculum network

### Step 3: Base Reward Design

-   Incorporates:
    -   Course features (out-degree, centrality, level)
    -   Expected GPA
    -   Match with student interests

### Step 4: Priority-based Reward Function

-   Customize reward based on student goals:
    -   Maximize GPA
    -   Graduate faster
    -   Follow interests
-   Use weighted combination of metrics per student

### Step 5: Q-Learning Framework

-   **State**: Passed courses, grades, major, program rules
-   **Action**: Select valid next-semester courses
-   **Reward**: From custom reward function
-   **Goal**: Maximize cumulative reward over course sequence

### Step 6: Model Training

-   Iterate across episodes to train agent
-   Output policy that maps states to optimal course selections

---

## Evaluation Methods

-   **Performance Metrics**: Total GPA, interest alignment score
-   **Baselines**:
    -   Random planner
    -   Fixed curriculum plan
-   **Q-Table Visualization**: Inspect high-value actions matching student priorities

---

## 📁 Repository Structure

````
PersonalizedAdvisor
├── data/
│   ├── semester_data/             # Raw dataset
│   ├── dependency_trees/
├── eda.py                      # Exploratory data analysis
├── graph_database.py           # Building the CPN
script
├── q_learning/
│   ├── agent.py                   # Q-learning agent logic
│   ├── reward_function.py         # Custom reward functions
│   ├── graph_utils.py             # Analyze the CPN features
│   └── train.py                   # Training loop
├── gpa_prediction/
│   ├── model.py
│   └── train.py                   # Training loop

├── notebooks/                  # Optional Jupyter notebooks for testing
├── README.md                   # Project overview
└── requirements.txt 
````

