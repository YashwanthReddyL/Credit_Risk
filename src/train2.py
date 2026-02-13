# ==========================================================
# CREDIT RISK PREDICTION - DAY 2
# Cross Validation + Model Improvement
# ==========================================================

# ==============================
# 1️⃣ IMPORT LIBRARIES
# ==============================

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ==============================
# 2️⃣ LOAD DATA
# ==============================

df = pd.read_csv("data/credit.csv")

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Dataset Shape:", df.shape)


# ==============================
# 3️⃣ DEFINE FEATURES & TARGET
# ==============================

y = df["SeriousDlqin2yrs"]
X = df.drop("SeriousDlqin2yrs", axis=1)

print("\nClass Distribution (%):")
class_dist = y.value_counts(normalize=True) * 100
print(class_dist.round(2))


# ==============================
# 4️⃣ TRAIN-TEST SPLIT (STRATIFIED)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 5️⃣ PREPROCESSING PIPELINE
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


# ==============================
# 6️⃣ CROSS VALIDATION SETUP
# ==============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==============================
# 7️⃣ LOGISTIC REGRESSION PIPELINE
# ==============================

lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

lr_cv_scores = cross_val_score(
    lr_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n===== CROSS VALIDATION (Logistic Regression) =====")
print(f"Mean ROC-AUC: {lr_cv_scores.mean() * 100:.2f}%")
print(f"Std Dev     : {lr_cv_scores.std() * 100:.2f}%")


# ==============================
# 8️⃣ RANDOM FOREST PIPELINE
# ==============================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n===== CROSS VALIDATION (Random Forest) =====")
print(f"Mean ROC-AUC: {rf_cv_scores.mean() * 100:.2f}%")
print(f"Std Dev     : {rf_cv_scores.std() * 100:.2f}%")


# ==============================
# 9️⃣ SELECT BEST MODEL
# ==============================

if rf_cv_scores.mean() > lr_cv_scores.mean():
    print("\n✅ Random Forest selected as best model.")
    best_model = rf_pipeline
else:
    print("\n✅ Logistic Regression selected as best model.")
    best_model = lr_pipeline


# ==============================
# 🔟 TRAIN BEST MODEL ON TRAIN SET
# ==============================

best_model.fit(X_train, y_train)


# ==============================
# 1️⃣1️⃣ FINAL TEST EVALUATION
# ==============================

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== FINAL TEST PERFORMANCE =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC : {roc_auc * 100:.2f}%")


print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC :", round(roc_auc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

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