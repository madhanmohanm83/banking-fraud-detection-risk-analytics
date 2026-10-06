import pandas as pd
from pathlib import Path

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "banking_transactions.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD DATASET
# ============================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# ============================================
# 3. CHECK MISSING VALUES
# ============================================

print("\nMissing values before preprocessing:")
print(df.isnull().sum().sum())

# ============================================
# 4. REMOVE DUPLICATES
# ============================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()

# ============================================
# 5. CONVERT TARGET VARIABLE
# ============================================

df["fraud_flag"] = df["fraud_flag"].astype(int)

print("\nTarget variable:")
print(df["fraud_flag"].value_counts())

# ============================================
# 6. REMOVE TRANSACTION ID
# ============================================

# transaction_id is an identifier, not a useful
# predictive feature for the fraud model.

df = df.drop(columns=["transaction_id"])

print("\nRemoved transaction_id.")

# ============================================
# 7. IDENTIFY CATEGORICAL FEATURES
# ============================================

categorical_columns = [
    "payment_channel",
    "authentication_type"
]

print("\nCategorical columns:")
print(categorical_columns)

# ============================================
# 8. ONE-HOT ENCODING
# ============================================

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\nCategorical encoding completed.")

# ============================================
# 9. FINAL DATA CHECK
# ============================================

print("\nProcessed dataset shape:", df.shape)

print("\nProcessed columns:")

for column in df.columns:
    print("-", column)

print("\nMissing values after preprocessing:")
print(df.isnull().sum().sum())

# ============================================
# 10. SAVE PROCESSED DATASET
# ============================================

processed_path = PROCESSED_DIR / "processed_transactions.csv"

df.to_csv(processed_path, index=False)

print("\nProcessed dataset saved successfully!")
print(processed_path)

# ============================================
# 11. FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)