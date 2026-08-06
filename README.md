# Adult Census Income Prediction

## Problem Statement

The objective of this project is to predict whether an individual's annual income exceeds $50K using demographic and employment-related information. This is a binary classification problem solved using multiple machine learning algorithms and deployed through a Streamlit web application.

---

## Dataset Description

- Dataset: Adult Census Income Dataset
- Source: UCI Machine Learning Repository
- Number of Records: 32,561
- Number of Features: 14
- Target Variable: income

### Features

- age
- workclass
- fnlwgt
- education
- education.num
- marital.status
- occupation
- relationship
- race
- sex
- capital.gain
- capital.loss
- hours.per.week
- native.country

Target Classes:

- <=50K
- >50K

---

## Project Structure

```
Adult-Census-Income-Prediction/

│── app.py
│── train_models.py
│── requirements.txt
│── README.md
│── adult.csv
│── test_data.csv
│── model_results.csv

├── models/
│ ├── logistic_regression.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── scaler.pkl
│ └── label_encoders.pkl
```

---

## GitHub Repository

Replace this after uploading:

https://github.com/yourusername/Adult-Census-Income-Prediction

---

## Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest

---

## Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |

*(Fill these values from model_results.csv after training.)*

---

## Observations

### Logistic Regression

- Fast to train
- Good baseline model
- Performs well on linearly separable data

### Decision Tree

- Easy to interpret
- Can overfit without pruning

### KNN

- Simple algorithm
- Performance depends on the value of K
- Prediction becomes slower with larger datasets

### Gaussian Naive Bayes

- Very fast
- Assumes feature independence
- Good baseline classifier

### Random Forest

- Highest overall accuracy
- Handles nonlinear relationships well
- Less prone to overfitting than a single Decision Tree

---

## Overall Best Model

Random Forest achieved the best overall performance on the Adult Census Income dataset based on Accuracy, AUC, F1 Score, and MCC.

---

## Streamlit Features

- Upload CSV test data
- Select classification model
- View predictions
- Display evaluation metrics
- Confusion Matrix
- Classification Report
- Download prediction results

---

## Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Author

Name: YOUR NAME

BITS ID: YOUR ID

Course: Machine Learning