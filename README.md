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

https://github.com/2025AC05843/Adult-Census-Income-Prediction-

## Streamlit app link

https://jfdxshjhukj5zxnkvhhlqd.streamlit.app/
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
| Logistic Regression |0.830598375600861 |0.864157876 |0.755319149 |0.472703063 |0.581490581 |0.503076025 |
| Decision Tree |0.811039284 |0.746907896805583 |0.62082777036048 |0.619174434087882 |0.62 |0.494256535685438 |
| KNN |0.833250455826288 |0.861496229113774 |0.684798807749627 |0.611850865512649 |0.646272855133614 |0.539096795511174 |
| Naive Bayes |0.790817172219459 |0.838862977076691 |0.667130919220055 |0.318908122503328 |0.431531531531531 |0.355436117559085 |
| Random Forest |0.863583623404608 |0.91355203288134 |0.77181745396317 |0.641810918774966 |0.700836059614685 |0.617827047358319 |

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

Name: DHINESH.G

BITS ID: 2025AC05843

Course: Machine Learning