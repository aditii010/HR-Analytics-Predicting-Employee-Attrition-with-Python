Employee Attrition Prediction using Random Forest Classifier
📋 Project Overview

This project aims to predict employee attrition — whether an employee is likely to leave or stay — using machine learning techniques.
By analyzing various factors such as satisfaction level, average monthly hours, salary, promotions, and work accidents, the model identifies employees at high risk of leaving.

The project uses the Random Forest Classifier, an ensemble algorithm known for its robustness and accuracy. The goal is to help HR departments make data-driven decisions and improve employee retention strategies.

Objectives :

Analyze employee data and identify key factors influencing attrition.

Build a machine learning model to predict attrition with high accuracy.

Evaluate model performance using multiple metrics.

Provide insights to HR teams for retention planning.

Dataset Description :

Dataset name: Dataset01-Employee_Attrition.csv

Feature	Description
satisfaction_level	 :Employee satisfaction score (0–1)
last_evaluation :	Performance evaluation score (0–1)
number_project	: Number of projects handled
average_montly_hours :	Average working hours per month
time_spend_company	: Years spent in the company
Work_accident	:1 if had an accident at work
promotion_last_5years:	1 if promoted in last 5 years
Department	: Employee department (encoded)
salary	Salary level (encoded as 0=low, 1=medium, 2=high)
left	Target variable: 1 = left, 0 = stayed
⚙️ Project Workflow ->

1️⃣ Data Preprocessing

Removed duplicates

Checked and handled missing values

Label encoded categorical columns (Department, salary)

Split data into training (80%) and testing (20%) sets

Scaled numerical features using StandardScaler

2️⃣ Exploratory Data Analysis (EDA)

Visualized attrition distribution using bar and pie charts

Analyzed relationship between salary, promotions, and attrition

Used histograms to visualize feature distributions

3️⃣ Model Building

Used RandomForestClassifier from Scikit-learn:

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_scaled, y_train)
y_pred = rf_model.predict(x_test_scaled)

4️⃣ Model Evaluation

Confusion Matrix

[[1983   15]
 [  53  348]]


Accuracy: 0.97

Precision: 0.96

Recall: 0.91

F1 Score: 0.91

Metric	Score
Accuracy	97%
Precision	95.8%
Recall	91%
F1-score	0.91
5️⃣ Cross Validation

Applied K-Fold Cross Validation (cv=5) to ensure reliability:

from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, x_train_scaled, y_train, cv=5, scoring='accuracy')


Average CV Accuracy ≈ 96.8%

📊 Confusion Matrix Visualization

Top-left (1983): Correctly predicted “stayed” employees

Bottom-right (348): Correctly predicted “left” employees

Misclassifications are minimal

🔍 Feature Importance

Extracted using:

import pandas as pd
feature_importances = pd.Series(rf_model.feature_importances_, index=x.columns).sort_values(ascending=False)
print(feature_importances)


Top factors affecting attrition:

Satisfaction level

Average monthly hours

Number of projects

Time spent at the company

Salary

💾 Saving and Loading Model
import joblib
joblib.dump(rf_model, "employee_attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")


To reuse:

model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

🔮 Predicting for a New Employee
new_emp = pd.DataFrame({
    'satisfaction_level':[0.45],
    'last_evaluation':[0.70],
    'number_project':[4],
    'average_montly_hours':[230],
    'time_spend_company':[3],
    'Work_accident':[0],
    'promotion_last_5years':[0],
    'Department':[5],
    'salary':[0]
})
new_emp_scaled = scaler.transform(new_emp)
prediction = model.predict(new_emp_scaled)
print(prediction)


Output:
[1] → Employee likely to leave
[0] → Employee likely to stay

📈 Results & Insights

The Random Forest Classifier achieved 97% accuracy — indicating excellent predictive performance.

Employee satisfaction, workload, and salary are the strongest predictors of attrition.

The model can help HR teams identify high-risk employees early and reduce turnover through targeted actions.

🧰 Tools & Technologies

Python 3.x

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Joblib

Jupyter Notebook / Google Colab

📂 Repository Structure
├── Dataset01-Employee_Attrition.csv
├── Employee_Attrition_Prediction.ipynb
├── employee_attrition_model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt

🧑‍💻 Author


Aditi Sikarwar
3rd Year B.Tech | Thapar University
📧 aditi.sikarwar25@gmail.com
