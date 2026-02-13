# ==========================================================
# CREDIT RISK FINAL MODEL SELECTION
# Random Forest vs XGBoost with Threshold Comparison
# ==========================================================

# ==============================
# 1️⃣ IMPORT LIBRARIES
# ==============================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from xgboost import XGBClassifier


# ==============================
# 2️⃣ LOAD DATA
# ==============================

df = pd.read_csv("data/credit.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Dataset Shape:", df.shape)

y = df["SeriousDlqin2yrs"]
X = df.drop("SeriousDlqin2yrs", axis=1)

print("\nClass Distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))


# ==============================
# 3️⃣ TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 4️⃣ PREPROCESSING
# ==============================

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==============================
# 5️⃣ RANDOM FOREST
# ==============================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_cv = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n===== RANDOM FOREST CV =====")
print(f"Mean ROC-AUC: {rf_cv.mean() * 100:.2f}%")


# ==============================
# 6️⃣ XGBOOST
# ==============================

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

xgb_cv = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n===== XGBOOST CV =====")
print(f"Mean ROC-AUC: {xgb_cv.mean() * 100:.2f}%")


# ==============================
# 7️⃣ TRAIN BOTH MODELS
# ==============================

rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

rf_test_auc = roc_auc_score(y_test, rf_prob)
xgb_test_auc = roc_auc_score(y_test, xgb_prob)

print("\n===== TEST ROC-AUC =====")
print(f"Random Forest: {rf_test_auc * 100:.2f}%")
print(f"XGBoost      : {xgb_test_auc * 100:.2f}%")


# ==============================
# 8️⃣ THRESHOLD COMPARISON
# ==============================

thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

print("\n===== THRESHOLD COMPARISON =====")

best_model = None
best_score = 0
best_model_name = ""
best_threshold = None

for t in thresholds:

    print(f"\n--- Threshold: {t} ---")

    # Random Forest
    rf_pred = (rf_prob >= t).astype(int)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)

    print(f"RF  -> Precision: {rf_precision * 100:.2f}% | Recall: {rf_recall * 100:.2f}%")

    # XGBoost
    xgb_pred = (xgb_prob >= t).astype(int)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)

    print(f"XGB -> Precision: {xgb_precision * 100:.2f}% | Recall: {xgb_recall * 100:.2f}%")

    # Balanced score logic
    rf_score = rf_precision * rf_recall
    xgb_score = xgb_precision * xgb_recall

    if rf_score > best_score:
        best_score = rf_score
        best_model = rf_pipeline
        best_model_name = "Random Forest"
        best_threshold = t

    if xgb_score > best_score:
        best_score = xgb_score
        best_model = xgb_pipeline
        best_model_name = "XGBoost"
        best_threshold = t


# ==============================
# 9️⃣ SAVE BEST MODEL
# ==============================

if best_model is not None:
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/credit_risk_model.pkl")
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Best Threshold: {best_threshold}")
    print("✅ Model saved as model/credit_risk_model.pkl")
else:
    print("\n⚠ No suitable model found.")
