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