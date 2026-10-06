import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "banking_transactions.csv"
RESULTS_DIR = BASE_DIR / "results" / "graphs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 2. LOAD DATASET
# ============================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("=" * 60)

print("\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ============================================
# 3. BASIC DATA INFORMATION
# ============================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================
# 4. STATISTICAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").transpose())


# ============================================
# 5. TARGET DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("FRAUD DISTRIBUTION")
print("=" * 60)

fraud_counts = df["fraud_flag"].value_counts()

print(fraud_counts)

print("\nFraud percentages:")
print(df["fraud_flag"].value_counts(normalize=True) * 100)


# ============================================
# 6. FRAUD DISTRIBUTION GRAPH
# ============================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="fraud_flag"
)

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Flag")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fraud_distribution.png",
    dpi=300
)

plt.close()


# ============================================
# 7. TRANSACTION AMOUNT DISTRIBUTION
# ============================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="transaction_amount",
    bins=30,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "transaction_amount_distribution.png",
    dpi=300
)

plt.close()


# ============================================
# 8. FRAUD VS TRANSACTION AMOUNT
# ============================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="transaction_amount"
)

plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Transaction Amount")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fraud_vs_transaction_amount.png",
    dpi=300
)

plt.close()


# ============================================
# 9. CORRELATION ANALYSIS
# ============================================

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================
# 10. DEVICE RISK SCORE ANALYSIS
# ============================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="device_risk_score"
)

plt.title("Device Risk Score by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Device Risk Score")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "device_risk_by_fraud.png",
    dpi=300
)

plt.close()


# ============================================
# 11. ANOMALY SCORE ANALYSIS
# ============================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="anomaly_score"
)

plt.title("Anomaly Score by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Anomaly Score")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "anomaly_score_by_fraud.png",
    dpi=300
)

plt.close()


# ============================================
# 12. INTERNATIONAL TRANSACTIONS
# ============================================

international_fraud = pd.crosstab(
    df["international_transaction_flag"],
    df["fraud_flag"],
    normalize="index"
) * 100

print("\nInternational Transaction vs Fraud (%):")
print(international_fraud)


# ============================================
# 13. SUSPICIOUS IP ANALYSIS
# ============================================

suspicious_ip_fraud = pd.crosstab(
    df["suspicious_ip_flag"],
    df["fraud_flag"],
    normalize="index"
) * 100

print("\nSuspicious IP vs Fraud (%):")
print(suspicious_ip_fraud)


# ============================================
# 14. PAYMENT CHANNEL ANALYSIS
# ============================================

payment_fraud = pd.crosstab(
    df["payment_channel"],
    df["fraud_flag"],
    normalize="index"
) * 100

print("\nPayment Channel vs Fraud (%):")
print(payment_fraud)


# ============================================
# 15. FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nGraphs saved in:")
print(RESULTS_DIR)