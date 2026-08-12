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
    "Logistic Regression": pickle.load(
        open("models/logistic_regression.pkl", "rb")
    ),
    "Decision Tree": pickle.load(
        open("models/decision_tree.pkl", "rb")
    ),
    "KNN": pickle.load(
        open("models/knn.pkl", "rb")
    ),
    "Naive Bayes": pickle.load(
        open("models/naive_bayes.pkl", "rb")
    ),
    "Random Forest": pickle.load(
        open("models/random_forest.pkl", "rb")
    )
}

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

encoders = pickle.load(
    open("models/label_encoders.pkl", "rb")
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

# --------------------------------------------------
# Upload Test Dataset
# --------------------------------------------------

st.subheader("📁 Upload CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# Stop the app until a CSV is uploaded
if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Read uploaded CSV
df = pd.read_csv(uploaded_file)

st.success("✅ CSV file uploaded successfully!")

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.write(
    f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}"
)

# --------------------------------------------------
# Check Target Column
# --------------------------------------------------

if "income" not in df.columns:
    st.error(
        "❌ The uploaded CSV must contain an 'income' column."
    )
    st.stop()

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

for col in df.select_dtypes(
    include=["object", "string"]
).columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

df.replace("?", np.nan, inplace=True)

# Remove rows containing missing values
df.dropna(inplace=True)

# Check if data remains
if df.empty:
    st.error(
        "❌ No valid rows remain after data cleaning."
    )
    st.stop()

# --------------------------------------------------
# Target
# --------------------------------------------------

try:
    y = df["income"].astype(int)
except ValueError:
    st.error(
        "❌ The 'income' column must contain 0 and 1 values."
    )
    st.stop()

X = df.drop(columns=["income"])

# --------------------------------------------------
# Encode Categorical Columns
# --------------------------------------------------

for col in X.columns:

    if col in encoders:

        le = encoders[col]

        mapping = {
            cls: idx
            for idx, cls in enumerate(le.classes_)
        }

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

try:

    if model_name in [
        "Logistic Regression",
        "KNN"
    ]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    pred = model.predict(X_input)

except Exception as e:

    st.error(
        f"❌ Error while making predictions: {e}"
    )
    st.stop()

# --------------------------------------------------
# Prediction Probability
# --------------------------------------------------

if hasattr(model, "predict_proba"):

    prob = model.predict_proba(X_input)[:, 1]

else:

    prob = pred

# --------------------------------------------------
# Metrics
# --------------------------------------------------

st.subheader("📈 Evaluation Metrics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{accuracy_score(y, pred):.4f}"
)

c2.metric(
    "Precision",
    f"{precision_score(y, pred, zero_division=0):.4f}"
)

c3.metric(
    "Recall",
    f"{recall_score(y, pred, zero_division=0):.4f}"
)

c1.metric(
    "F1 Score",
    f"{f1_score(y, pred, zero_division=0):.4f}"
)

try:

    auc = roc_auc_score(y, prob)

except ValueError:

    auc = 0.0

c2.metric(
    "AUC",
    f"{auc:.4f}"
)

c3.metric(
    "MCC",
    f"{matthews_corrcoef(y, pred):.4f}"
)

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

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=12
        )

st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y,
    pred,
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True
)

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

st.dataframe(
    result.head(20),
    use_container_width=True
)
