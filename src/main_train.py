# ==========================================
# FINAL PRODUCTION TRAINING (FULL DATA)
# ==========================================

import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


# --------------------------
# Load Data
# --------------------------

df = pd.read_csv("data/credit.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

y = df["SeriousDlqin2yrs"]
X = df.drop("SeriousDlqin2yrs", axis=1)

print("Training on FULL dataset:", X.shape)


# --------------------------
# Preprocessing
# --------------------------

numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)

# --------------------------
# Final Selected Model
# --------------------------

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# --------------------------
# Train on FULL Data
# --------------------------

final_model.fit(X, y)

# --------------------------
# Save Model
# --------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(final_model, "model/credit_risk_model.pkl")

print("✅ Final production model trained on FULL data and saved.")
