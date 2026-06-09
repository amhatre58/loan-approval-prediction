## Loan Approval Prediction Using Machine Learning
This project builds a binary classification system to predict whether a loan application should be approved or rejected. It leverages applicant details such as income, credit history, loan amount, and employment status to simulate automated, real-world financial decision-making.
------------------------------
## 📌 Project Overview
Manual loan evaluation is time-consuming and prone to human error. This machine learning pipeline automates the process, helping financial institutions minimize risk (default rates) while maximizing approval efficiency for qualified applicants.
------------------------------
## ⚙️ Core Workflow## 1. Data Cleaning & Preprocessing

* Missing Value Imputation: Handling null inputs in critical fields like credit history and self-employment status.
* Categorical Encoding: Converting text variables (e.g., Education, Marital Status) into numerical formats via One-Hot or Label Encoding.
* Feature Scaling: Normalizing continuous inputs (e.g., Applicant Income, Loan Amount) to ensure balanced model training.

## 2. Exploratory Data Analysis (EDA)

* Analyzing the definitive impact of Credit History on final loan status.
* Visualizing income distributions and identifying outliers that skew predictions.
* Checking for class imbalances between approved and rejected applications.

## 3. Model Training & Evaluation
The project implements and compares multiple classification strategies:

* Baseline Model: Logistic Regression for interpretability.
* Tree-Based Models: Decision Trees and Random Forests to capture non-linear relationships.
* Evaluation Metrics: Accuracy, Precision, Recall, and F1-score, with a heavy emphasis on minimizing False Negatives (approving high-risk borrowers).

------------------------------
## 🚀 Getting Started## Prerequisites
Ensure you have Python 3.8+ installed along with the following libraries:

pip install numpy pandas matplotlib seaborn scikit-learn

## 📊 Expected Performance
The final models are evaluated to ensure data-driven risk management:

* High Recall: Ensures fewer risky loans are accidentally approved.
* High Precision: Ensures eligible applicants are not unfairly rejected.
