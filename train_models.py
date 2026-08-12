import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("adult.csv")

print(df.head())
print(df.dtypes)
print(df.columns)

# Remove leading/trailing spaces from all string columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip()

# Replace missing values
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# -------------------------------
# Encode categorical columns
# -------------------------------
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

# Encode all string/object columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
# -------------------------------
# Features and Target
# -------------------------------
X = df.drop("income", axis=1)
y = df["income"]

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

results = []

# -------------------------------
# Training Loop
# -------------------------------
for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Accuracy :", accuracy)
    print("AUC :", auc)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 :", f1)
    print("MCC :", mcc)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    filename = (
        "models/"
        + name.lower().replace(" ", "_")
        + ".pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

# Save label encoders
pickle.dump(label_encoders, open("models/label_encoders.pkl", "wb"))

# Save results
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

results_df.to_csv("model_results.csv", index=False)

print("\nTraining Completed Successfully!")
print(results_df)

# Save test dataset
test_data = X_test.copy()
test_data["income"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("test_data.csv saved successfully!")

