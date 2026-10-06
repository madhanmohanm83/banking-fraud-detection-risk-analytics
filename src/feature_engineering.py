import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "processed_transactions.csv"
)

FEATURE_DIR = BASE_DIR / "data" / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD PROCESSED DATASET
# ============================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nProcessed dataset shape:", df.shape)

# ============================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================

X = df.drop(columns=["fraud_flag"])
y = df["fraud_flag"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# ============================================
# 4. CREATE RISK-RELATED FEATURES
# ============================================

# Combined behavioral risk indicators

X["behavior_risk_score"] = (
    X["login_attempts"] * 2
    + X["failed_transactions_last_30d"] * 2
    + X["suspicious_ip_flag"] * 5
    + X["international_transaction_flag"] * 2
)

# Transaction activity intensity

X["transaction_activity_score"] = (
    X["daily_transaction_count"]
    + X["transfer_frequency"]
    + X["transaction_velocity_score"]
)

# Account age category feature

X["new_account_flag"] = (
    X["account_age_days"] < 365
).astype(int)

# High transaction amount indicator

amount_threshold = X["transaction_amount"].quantile(0.75)

X["high_transaction_amount_flag"] = (
    X["transaction_amount"] > amount_threshold
).astype(int)

print("\nFeature engineering completed.")

print("\nNew features:")
print("- behavior_risk_score")
print("- transaction_activity_score")
print("- new_account_flag")
print("- high_transaction_amount_flag")

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ============================================
# 6. SAVE FEATURE DATA
# ============================================

X_train.to_csv(
    FEATURE_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    FEATURE_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    FEATURE_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    FEATURE_DIR / "y_test.csv",
    index=False
)

print("\nFeature datasets saved successfully!")

print(FEATURE_DIR)

# ============================================
# 7. FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 60)