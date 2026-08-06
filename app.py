import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Adult Census Income Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Adult Census Income Prediction")
st.write("Machine Learning Assignment - Adult Census Income Dataset")

# --------------------------------------------------
# Load Models
# --------------------------------------------------
models = {
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("models/knn.pkl", "rb")),
    "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb"))
}

scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

# --------------------------------------------------
# Load Test Dataset Automatically
# --------------------------------------------------
df = pd.read_csv("test_data.csv")

st.success("✅ test_data.csv loaded successfully")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# --------------------------------------------------
# Target
# --------------------------------------------------
y = df["income"].astype(int)
X = df.drop(columns=["income"])

# --------------------------------------------------
# Encode Categorical Columns
# --------------------------------------------------
for col in X.columns:
    if col in encoders:
        le = encoders[col]
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

        X[col] = (
            X[col]
            .astype(str)
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
model = models[model_name]

if model_name in ["Logistic Regression", "KNN"]:
    X_input = scaler.transform(X)
else:
    X_input = X

pred = model.predict(X_input)

if hasattr(model, "predict_proba"):
    prob = model.predict_proba(X_input)[:, 1]
else:
    prob = pred

# --------------------------------------------------
# Metrics
# --------------------------------------------------
st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
c2.metric("Precision", f"{precision_score(y, pred):.4f}")
c3.metric("Recall", f"{recall_score(y, pred):.4f}")

c1.metric("F1 Score", f"{f1_score(y, pred):.4f}")
c2.metric("AUC", f"{roc_auc_score(y, prob):.4f}")
c3.metric("MCC", f"{matthews_corrcoef(y, pred):.4f}")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, pred)

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_xticklabels(["<=50K", ">50K"])

ax.set_yticks([0, 1])
ax.set_yticklabels(["<=50K", ">50K"])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("Classification Report")

report = classification_report(
    y,
    pred,
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())

# --------------------------------------------------
# Predictions
# --------------------------------------------------
st.subheader("Prediction Results")

result = df.copy()

result["Prediction"] = np.where(
    pred == 1,
    ">50K",
    "<=50K"
)

st.dataframe(result.head(20))

# --------------------------------------------------
# Model Performance Table
# --------------------------------------------------
st.subheader("Model Performance Comparison")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [0.8306, 0.8110, 0.8333, 0.7908, 0.8636],
    "AUC": [0.8642, 0.7469, 0.8615, 0.8389, 0.9136],
    "Precision": [0.7553, 0.6208, 0.6848, 0.6671, 0.7718],
    "Recall": [0.4727, 0.6192, 0.6119, 0.3189, 0.6418],
    "F1 Score": [0.5815, 0.6200, 0.6463, 0.4315, 0.7008],
    "MCC": [0.5031, 0.4943, 0.5391, 0.3554, 0.6178]
})

st.dataframe(performance, use_container_width=True)

# --------------------------------------------------
# Observations
# --------------------------------------------------
st.subheader("Model Observation")

observations = {
    "Logistic Regression":
        "Accuracy: 83.06%. Good precision but lower recall. Performs well as a simple baseline model.",

    "Decision Tree":
        "Accuracy: 81.10%. Balanced precision and recall but prone to overfitting.",

    "KNN":
        "Accuracy: 83.33%. Better recall than Logistic Regression with a balanced F1-score.",

    "Naive Bayes":
        "Accuracy: 79.08%. Lowest-performing model with the lowest recall and F1-score.",

    "Random Forest":
        "Accuracy: 86.36%. Best overall performance with the highest Accuracy, AUC, F1-score and MCC."
}

st.info(observations[model_name])

# --------------------------------------------------
# Overall Winner
# --------------------------------------------------
st.success("""
🏆 Overall Winner: Random Forest

Reasons:
- Highest Accuracy (86.36%)
- Highest AUC (0.9136)
- Highest F1 Score (0.7008)
- Highest MCC (0.6178)

Random Forest achieved the best overall performance on the Adult Census Income dataset.
""")