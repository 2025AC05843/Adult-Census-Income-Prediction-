**Adult Census Income Prediction**

**Problem Statement**

The objective of this project is to predict whether an individual's annual income exceeds $50K using demographic and employment-related information. This is a binary classification problem solved using multiple machine learning algorithms and deployed through a Streamlit web application.

**Dataset Description**

Dataset: Adult Census Income Dataset

Source: UCI Machine Learning Repository

Number of Records: 32,561

Number of Features: 14

Target Variable: income

**Features**

age

workclass

fnlwgt

education

education.num

marital.status

occupation

relationship

race

sex

capital.gain

capital.loss

hours.per.week

native.country

**Target Classes:**

<=50K
50K

**Project Structure**

Adult-Census-Income-Prediction/
│── app.py
│── train_models.py
│── requirements.txt
│── README.md
│── adult.csv
│── test_data.csv
│── model_results.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl

**GitHub Repository**

https://github.com/2025AC05843/Adult-Census-Income-Prediction-

**Streamlit app link**

https://jfdxshjhukj5zxnkvhhlqd.streamlit.app/

**Models Used**

Logistic Regression

Decision Tree

K-Nearest Neighbors

Gaussian Naive Bayes

Random Forest

**Evaluation Metrics**

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

**Model Results**

Model

Accuracy

AUC

Precision

Recall

F1 Score

MCC

Logistic Regression

0.8306

0.8642

0.7553

0.4727

0.5815

0.5031

Decision Tree

0.8110

0.7469

0.6208

0.6192

0.6200

0.4943

KNN

0.8333

0.8615

0.6848

0.6119

0.6463

0.5391

Naive Bayes

0.7908

0.8389

0.6671

0.3189

0.4315

0.3554

Random Forest

0.8636

0.9136

0.7718

0.6418

0.7008

0.6178

**Observations**

| **ML Model Name** | **Observation about Model Performance** |
|---|---|
| **Logistic Regression** | Achieved **83.06% accuracy** and **0.8642 AUC**. It has good precision (0.7553) but relatively low recall (0.4727), making it a strong baseline model. |
| **Decision Tree** | Achieved **81.10% accuracy** and **0.7469 AUC**. It has better recall (0.6192) than Logistic Regression but lower precision and overall performance. |
| **kNN** | Achieved **83.33% accuracy** and **0.8615 AUC**. It provides balanced performance with an F1 score of **0.6463** and MCC of **0.5391**. |
| **Naive Bayes** | Achieved **79.08% accuracy** and **0.8389 AUC**. It has the lowest recall (**0.3189**) and F1 score (**0.4315**), making it the weakest overall performer. |
| **Random Forest (Ensemble)** | Achieved the **highest performance across all metrics**: 86.36% accuracy, 0.9136 AUC, 0.7718 precision, 0.6418 recall, 0.7008 F1, and 0.6178 MCC. |
| **Overall Winner** | **Random Forest (Ensemble)** is the overall winner because it performs best across all six evaluation metrics. |

**Streamlit Features**

Select classification model

View predictions

Display evaluation metrics

Confusion Matrix

Classification Report

Download prediction results

**Requirements**

Python 3.10+

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

**Author**

Name: DHINESH.G

BITS ID: 2025AC05843

Course: Machine Learning
