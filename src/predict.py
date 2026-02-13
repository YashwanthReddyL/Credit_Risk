import joblib
import pandas as pd

# Load trained full-data model
model = joblib.load("model/credit_risk_model.pkl")

threshold = 0.2

new_customer = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
}])

probability = model.predict_proba(new_customer)[:, 1][0]

prediction = 1 if probability >= threshold else 0

print(f"Default Probability: {probability * 100:.2f}%")
print("Prediction:", "High Risk" if prediction == 1 else "Low Risk")
