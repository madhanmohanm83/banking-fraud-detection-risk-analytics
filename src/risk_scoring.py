import pandas as pd
import joblib
from pathlib import Path

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
RESULTS_DIR = BASE_DIR / "results" / "risk_analysis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD TEST DATA
# ============================================

X_test = pd.read_csv(
    FEATURE_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    FEATURE_DIR / "y_test.csv"
).squeeze()

print("=" * 60)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("RISK SCORING")
print("=" * 60)

print("\nTest data:", X_test.shape)

# ============================================
# 3. LOAD TRAINED MODEL
# ============================================

model = joblib.load(MODEL_PATH)

print("\nRandom Forest model loaded successfully.")

# ============================================
# 4. GENERATE FRAUD PROBABILITY
# ============================================

fraud_probability = model.predict_proba(X_test)[:, 1]

# Convert probability into 0-100 risk score
risk_score = fraud_probability * 100

# ============================================
# 5. CREATE RISK LEVEL
# ============================================

def get_risk_level(score):

    if score < 25:
        return "Low Risk"

    elif score < 50:
        return "Medium Risk"

    elif score < 75:
        return "High Risk"

    else:
        return "Critical Risk"


risk_level = [
    get_risk_level(score)
    for score in risk_score
]

# ============================================
# 6. CREATE RISK RESULTS DATAFRAME
# ============================================

risk_results = X_test.copy()

risk_results["actual_fraud"] = y_test.values

risk_results["fraud_probability"] = fraud_probability

risk_results["risk_score"] = risk_score

risk_results["risk_level"] = risk_level

# ============================================
# 7. DISPLAY RISK DISTRIBUTION
# ============================================

print("\nRisk Level Distribution:")

print(
    risk_results["risk_level"]
    .value_counts()
    .sort_index()
)

# ============================================
# 8. DISPLAY SAMPLE RESULTS
# ============================================

print("\nSample Risk Results:")

print(
    risk_results[
        [
            "fraud_probability",
            "risk_score",
            "risk_level",
            "actual_fraud"
        ]
    ].head(10)
)

# ============================================
# 9. SAVE RISK RESULTS
# ============================================

output_path = (
    RESULTS_DIR
    / "transaction_risk_scores.csv"
)

risk_results.to_csv(
    output_path,
    index=False
)

print("\nRisk scoring results saved successfully!")

print(output_path)

# ============================================
# 10. FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("RISK SCORING COMPLETED SUCCESSFULLY!")
print("=" * 60)