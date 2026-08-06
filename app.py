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

st.set_page_config(
    page_title="Adult Census Income Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Prediction")
st.write("Upload a CSV file to evaluate or predict income.")

# ----------------------------
# Load Models
# ----------------------------
models = {
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("models/knn.pkl", "rb")),
    "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb"))
}

scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

# ----------------------------
# Sidebar
# ----------------------------
model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ----------------------------
    # Clean Data
    # ----------------------------
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
        )

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    # ----------------------------
    # Check if target exists
    # ----------------------------
    has_target = "income" in df.columns

    if has_target:
        income_mapping = {
            "<=50K": 0,
            ">50K": 1
        }

        y = (
            df["income"]
            .map(income_mapping)
        )

        if y.isnull().any():
            st.error("Unknown values found in the income column.")
            st.stop()

        y = y.astype(int)

        X = df.drop(columns=["income"])

    else:
        X = df.copy()

    # ----------------------------
    # Encode categorical columns
    # ----------------------------
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

    # ----------------------------
    # Scaling
    # ----------------------------
    model = models[model_name]

    if model_name in ["Logistic Regression", "KNN"]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    predictions = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_input)[:, 1]
    else:
        probabilities = predictions

    prediction_labels = np.where(
        predictions == 0,
        "<=50K",
        ">50K"
    )

    # ----------------------------
    # Metrics
    # ----------------------------
    if has_target:

        st.subheader("Evaluation Metrics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{accuracy_score(y, predictions):.4f}"
        )

        c2.metric(
            "Precision",
            f"{precision_score(y, predictions):.4f}"
        )

        c3.metric(
            "Recall",
            f"{recall_score(y, predictions):.4f}"
        )

        c1.metric(
            "F1 Score",
            f"{f1_score(y, predictions):.4f}"
        )

        c2.metric(
            "AUC",
            f"{roc_auc_score(y, probabilities):.4f}"
        )

        c3.metric(
            "MCC",
            f"{matthews_corrcoef(y, predictions):.4f}"
        )

        # ----------------------------
        # Confusion Matrix
        # ----------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, predictions)

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["<=50K", ">50K"])
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

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose()
        )

    # ----------------------------
    # Prediction Results
    # ----------------------------
    result = df.copy()

    result["Prediction"] = prediction_labels

    st.subheader("Predictions")

    st.dataframe(result)

    st.download_button(
        "Download Predictions",
        result.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )

else:
    st.info("Upload test_data.csv or any Adult Census dataset CSV.")