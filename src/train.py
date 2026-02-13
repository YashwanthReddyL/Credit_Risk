# ==========================================
# CREDIT RISK PREDICTION - TRAINING SCRIPT
# ==========================================

# 1️⃣ IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# 2️⃣ LOAD DATA
df = pd.read_csv("data/credit.csv")

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Dataset Shape:", df.shape)


# 3️⃣ DEFINE FEATURES & TARGET
y = df["SeriousDlqin2yrs"]
X = df.drop("SeriousDlqin2yrs", axis=1)


# 4️⃣ TRAIN-TEST SPLIT (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nClass Distribution in %:")
print(y.value_counts(normalize=True) *100)


# 5️⃣ PREPROCESSING PIPELINE

# All features are numerical in this dataset
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


# 6️⃣ BUILD FULL ML PIPELINE

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# 7️⃣ TRAIN MODEL
pipeline.fit(X_train, y_train)


# 8️⃣ EVALUATE MODEL

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC :", round(roc_auc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# END OF DAY 1 BASELINE MODEL
# ==========================================
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n===== THRESHOLD ANALYSIS =====")

thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

for t in thresholds:
    y_pred_custom = (y_prob >= t).astype(int)
    
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    
    print(f"\nThreshold: {t}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
