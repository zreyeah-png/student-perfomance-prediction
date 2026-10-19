# 🎓 Student Performance Prediction using Machine Learning

A beginner-friendly Machine Learning project built using Python and Scikit-Learn that predicts whether a student will **Pass or Fail** their final Mathematics exam based on demographic, social, and academic factors.

---

## 📌 1. Project Overview & The Idea

### What does this model predict?
This project implements binary classification to predict student academic outcomes:
- **Pass (`1`):** Final Grade ($G3$) $\ge 10$ out of 20.
- **Fail (`0`):** Final Grade ($G3$) $< 10$ out of 20.

### Why is this useful?
- **Early Warning System:** Flags students at risk of academic failure before final exams take place.
- **Actionable Guidance:** Enables schools, teachers, and academic counselors to provide timely tutoring, mentorship, and personalized resources.
- **Factor Discovery:** Identifies the behavioral and academic factors (like prior grades, failures, and study time) that have the highest correlation with academic success.

---

## 📊 2. Dataset Information
- **Dataset Name:** Student Performance Dataset
- **Source:** [UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008)
- **Course Subject:** Mathematics (`student-mat.csv`)
- **Instances:** 395 students
- **Features:** 33 attributes (including study time, family background, past failures, absences, and period grades)

---

## ⚙️ 3. Setup and Installation Instructions

### Prerequisites
Make sure you have Python 3.8+ installed on your computer.

### Step 1: Clone or Download this Repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook student_performance_prediction.ipynb
```
*(Or open the repository in VS Code and run the notebook cells directly).*

---

## 🛠️ 4. Data Preparation & Models Used

1. **Data Cleaning:** Verified zero missing values across all 33 columns.
2. **Feature Engineering:** Converted final grade $G3$ into a binary target (`Passed`: 1 if $\ge 10$, else 0).
3. **One-Hot Encoding:** Converted all categorical features into numerical indicators using `pd.get_dummies`.
4. **Train-Test Split:** Stratified 80% training set (316 students) and 20% test set (79 students) with `random_state=42`.
5. **Feature Scaling:** Applied `StandardScaler` (mean=0, variance=1) for distance-based models.

### Models Implemented:
- **Decision Tree Classifier:** Flowchart-based rule model splitting on high-information features (`max_depth=4`).
- **K-Nearest Neighbors (KNN):** Distance-based algorithm classifying students based on the majority label of their $K=5$ most similar peers.

---

## 📈 5. Results & Evaluation

| Model | Test Accuracy | Precision (Fail / Pass) | Recall (Fail / Pass) |
|---|---|---|---|
| **Decision Tree** | **87.34%** | **0.74 / 0.98** | **0.96 / 0.83** |
| **K-Nearest Neighbors (KNN)** | **69.62%** | **0.57 / 0.72** | **0.31 / 0.89** |

### Confusion Matrix Highlights:
- **Decision Tree:** Correctly identified **25 out of 26 failing students** (96% Recall for the Fail class).
- **KNN:** Correctly identified only **8 out of 26 failing students**, misclassifying 18 at-risk students as passing.

---

## 🏆 6. Conclusion & What We Learned

1. **Best Performing Model:** **Decision Tree** significantly outperformed KNN (87.34% vs 69.62%).
2. **Why Decision Tree Won:**
   - Academic progression has strong hierarchical logic (e.g. earlier test scores `G2` and previous `failures` are direct indicators of passing).
   - KNN suffered from the *curse of dimensionality* due to the 41 features resulting from one-hot encoding.
3. **Key Takeaway:** For early warning systems, **Recall on the at-risk class** is far more critical than raw overall accuracy. Missing an at-risk student (False Positive) carries a much higher real-world cost than giving extra tutoring to a student who might have passed anyway.

---

## 📁 Repository Structure
```
student-performance-prediction/
│
├── data/
│   └── student-mat.csv              # Official UCI Mathematics dataset
├── student_performance_prediction.ipynb   # Complete, runnable Jupyter Notebook
├── README.md                        # Project documentation and summary
└── requirements.txt                 # Python dependencies
```
