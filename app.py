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

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Adult Census Income Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Prediction")
st.write("Machine Learning Assignment")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
models = {
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("models/knn.pkl", "rb")),
    "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb"))
}

scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------------------------------------------
    # Read CSV
    # ---------------------------------------------------
    df = pd.read_csv(uploaded_file)

    # Clean string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    if "income" not in df.columns:
        st.error("Uploaded CSV must contain the 'income' column.")
        st.stop()

    # ---------------------------------------------------
    # Target
    # ---------------------------------------------------
    y = encoders["income"].transform(df["income"].astype(str))

    X = df.drop(columns=["income"])

    # ---------------------------------------------------
    # Encode categorical columns
    # ---------------------------------------------------
    for col in X.columns:

        if col in encoders:

            le = encoders[col]

            X[col] = X[col].astype(str)

            mapping = dict(
                zip(
                    le.classes_,
                    le.transform(le.classes_)
                )
            )

            X[col] = X[col].map(mapping)

            # Unknown category -> first encoded class
            X[col] = X[col].fillna(0).astype(int)

    # ---------------------------------------------------
    # Select model
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    mcc = matthews_corrcoef(y, predictions)

    st.subheader("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")

    c1.metric("F1 Score", f"{f1:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("MCC", f"{mcc:.4f}")

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(5,4))

    im = ax.imshow(cm)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticklabels(["<=50K", ">50K"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12)

    st.pyplot(fig)

    # ---------------------------------------------------
    # Classification Report
    # ---------------------------------------------------
    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------
    result = df.copy()

    result["Prediction"] = encoders["income"].inverse_transform(
        predictions
    )

    st.subheader("Prediction Results")

    st.dataframe(result)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload the generated test_data.csv file to evaluate the models.")