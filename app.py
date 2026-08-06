import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,confusion_matrix,classification_report

st.set_page_config(page_title="Adult Census Income Prediction",page_icon="📊",layout="wide")
st.title("Adult Census Income Prediction")

models={
"Logistic Regression":pickle.load(open("models/logistic_regression.pkl","rb")),
"Decision Tree":pickle.load(open("models/decision_tree.pkl","rb")),
"KNN":pickle.load(open("models/knn.pkl","rb")),
"Naive Bayes":pickle.load(open("models/naive_bayes.pkl","rb")),
"Random Forest":pickle.load(open("models/random_forest.pkl","rb"))
}
scaler=pickle.load(open("models/scaler.pkl","rb"))
encoders=pickle.load(open("models/label_encoders.pkl","rb"))

model_name=st.sidebar.selectbox("Select Model",list(models.keys()))
uploaded=st.file_uploader("Upload CSV",type="csv")

if uploaded:
    df=pd.read_csv(uploaded)
    for c in df.select_dtypes(include=["object","string"]).columns:
        df[c]=df[c].astype(str).str.strip().str.replace(".","",regex=False)
    df.replace("?",np.nan,inplace=True)
    df.dropna(inplace=True)
    has_target="income" in df.columns
    if has_target:
        if pd.api.types.is_numeric_dtype(df["income"]):
            y=df["income"].astype(int)
        else:
            m={"<=50K":0,">50K":1}
            y=df["income"].map(m)
            if y.isnull().any():
                st.error(f"Unsupported income values: {df['income'].unique()}")
                st.stop()
            y=y.astype(int)
        X=df.drop(columns=["income"])
    else:
        X=df.copy()

    for c in X.columns:
        if c in encoders:
            le=encoders[c]
            mp={v:i for i,v in enumerate(le.classes_)}
            X[c]=X[c].astype(str).map(mp).fillna(0).astype(int)

    model=models[model_name]
    Xin=scaler.transform(X) if model_name in ["Logistic Regression","KNN"] else X
    pred=model.predict(Xin)
    probs=model.predict_proba(Xin)[:,1] if hasattr(model,"predict_proba") else pred

    if has_target:
        st.write({
            "Accuracy":accuracy_score(y,pred),
            "Precision":precision_score(y,pred),
            "Recall":recall_score(y,pred),
            "F1":f1_score(y,pred),
            "AUC":roc_auc_score(y,probs),
            "MCC":matthews_corrcoef(y,pred)
        })
        st.text(classification_report(y,pred))
        fig,ax=plt.subplots()
        ax.imshow(confusion_matrix(y,pred))
        st.pyplot(fig)

    out=df.copy()
    out["Prediction"]=np.where(pred==1,">50K","<=50K")
    st.dataframe(out.head())
    st.download_button("Download Predictions",out.to_csv(index=False),"predictions.csv","text/csv")


    # -----------------------------------------
# Model Performance & Observations
# -----------------------------------------

st.subheader("Model Performance & Observations")

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

st.markdown("### Model Observations")

observations = {
    "Logistic Regression":
        "Accuracy: 83.06%. Good precision but lower recall. Performs well as a simple baseline model.",

    "Decision Tree":
        "Accuracy: 81.10%. Balanced precision and recall, but more susceptible to overfitting.",

    "KNN":
        "Accuracy: 83.33%. Better recall than Logistic Regression with a balanced F1-score, but slower on large datasets.",

    "Naive Bayes":
        "Accuracy: 79.08%. Lowest-performing model with the lowest recall and F1-score.",

    "Random Forest":
        "Accuracy: 86.36%. Best overall performance with the highest AUC, F1-score, and MCC."
}

st.info(observations[model_name])

st.success(
    """
🏆 **Overall Winner: Random Forest**

**Reason**

- Highest Accuracy (86.36%)
- Highest AUC (0.9136)
- Highest F1 Score (0.7008)
- Highest MCC (0.6178)

Random Forest provided the best balance between precision and recall, making it the most suitable model for the Adult Census Income dataset.
"""
)