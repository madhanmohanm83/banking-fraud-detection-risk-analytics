import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Fraud Detection & Risk Analytics",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "banking_transactions.csv"
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "processed_transactions.csv"
)

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.csv"

RISK_PATH = (
    BASE_DIR / "results" / "risk_analysis" / "transaction_risk_scores.csv"
)

MODEL_COMPARISON_PATH = (
    BASE_DIR / "results" / "model_evaluation" / "model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names():
    df = pd.read_csv(FEATURE_NAMES_PATH)
    return df.iloc[:, 0].tolist()


@st.cache_data
def load_risk_data():
    return pd.read_csv(RISK_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:
    df = load_data()
    processed_df = load_processed_data()
    model = load_model()
    feature_names = load_feature_names()
    risk_df = load_risk_data()
    model_comparison = load_model_comparison()

except Exception as e:
    st.error("Project files could not be loaded.")
    st.error(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Fraud Detection")

st.sidebar.write(
    "Banking Fraud Detection & Risk Analytics"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard Overview",
        "Fraud Analytics",
        "Risk Analysis",
        "Transaction Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Machine Learning based banking transaction "
    "fraud detection system."
)


# ============================================================
# PAGE 1 - DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.title("🏦 Banking Fraud Detection & Risk Analytics")

    st.write(
        "Interactive dashboard for analyzing banking transactions "
        "and identifying potentially fraudulent transactions."
    )

    st.markdown("---")

    # Metrics
    total_transactions = len(df)
    fraud_transactions = int(df["fraud_flag"].sum())
    non_fraud_transactions = total_transactions - fraud_transactions
    fraud_percentage = (
        fraud_transactions / total_transactions * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Non-Fraud Transactions",
        f"{non_fraud_transactions:,}"
    )

    col4.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )

    st.markdown("---")

    # Fraud distribution
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Fraud Distribution")

        fraud_counts = df["fraud_flag"].value_counts()

        fraud_labels = ["Non-Fraud", "Fraud"]

        values = [
            fraud_counts.get(False, fraud_counts.get(0, 0)),
            fraud_counts.get(True, fraud_counts.get(1, 0))
        ]

        fig, ax = plt.subplots()

        ax.bar(fraud_labels, values)

        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Fraud vs Non-Fraud Transactions")

        st.pyplot(fig)

    with col2:

        st.subheader("Transaction Amount Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["transaction_amount"],
            bins=30
        )

        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Frequency")
        ax.set_title("Transaction Amount Distribution")

        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Recent Transactions")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 2 - FRAUD ANALYTICS
# ============================================================

elif page == "Fraud Analytics":

    st.title("📊 Fraud Analytics")

    st.write(
        "Explore transaction patterns and fraud-related behavior."
    )

    st.markdown("---")

    # Payment Channel Analysis
    st.subheader("Fraud by Payment Channel")

    payment_analysis = (
        df.groupby("payment_channel")["fraud_flag"]
        .agg(["count", "sum"])
        .reset_index()
    )

    payment_analysis.columns = [
        "Payment Channel",
        "Total Transactions",
        "Fraud Transactions"
    ]

    st.dataframe(
        payment_analysis,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        payment_analysis["Payment Channel"],
        payment_analysis["Fraud Transactions"]
    )

    ax.set_xlabel("Payment Channel")
    ax.set_ylabel("Fraud Transactions")
    ax.set_title("Fraud Transactions by Payment Channel")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    st.markdown("---")

    # Authentication Analysis
    st.subheader("Fraud by Authentication Type")

    auth_analysis = (
        df.groupby("authentication_type")["fraud_flag"]
        .agg(["count", "sum"])
        .reset_index()
    )

    auth_analysis.columns = [
        "Authentication Type",
        "Total Transactions",
        "Fraud Transactions"
    ]

    st.dataframe(
        auth_analysis,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        auth_analysis["Authentication Type"],
        auth_analysis["Fraud Transactions"]
    )

    ax.set_xlabel("Authentication Type")
    ax.set_ylabel("Fraud Transactions")
    ax.set_title("Fraud Transactions by Authentication Type")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    st.markdown("---")

    # Transaction amount comparison
    st.subheader("Transaction Amount vs Fraud")

    fig, ax = plt.subplots()

    df.boxplot(
        column="transaction_amount",
        by="fraud_flag",
        ax=ax
    )

    ax.set_xlabel("Fraud Flag")
    ax.set_ylabel("Transaction Amount")
    ax.set_title("Transaction Amount by Fraud Status")

    plt.suptitle("")

    st.pyplot(fig)


# ============================================================
# PAGE 3 - RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title("⚠️ Transaction Risk Analysis")

    st.write(
        "Transactions are assigned risk scores based on the "
        "fraud probability predicted by the machine learning model."
    )

    st.markdown("---")

    # Risk metrics
    risk_counts = risk_df["risk_level"].value_counts()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Low Risk",
        risk_counts.get("Low Risk", 0)
    )

    col2.metric(
        "Medium Risk",
        risk_counts.get("Medium Risk", 0)
    )

    col3.metric(
        "High Risk",
        risk_counts.get("High Risk", 0)
    )

    col4.metric(
        "Critical Risk",
        risk_counts.get("Critical Risk", 0)
    )

    st.markdown("---")

    st.subheader("Risk Level Distribution")

    fig, ax = plt.subplots()

    risk_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Risk Distribution")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Risk Score Data")

    risk_filter = st.multiselect(
        "Filter Risk Level",
        options=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Critical Risk"
        ],
        default=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Critical Risk"
        ]
    )

    filtered_risk = risk_df[
        risk_df["risk_level"].isin(risk_filter)
    ]

    st.dataframe(
        filtered_risk,
        use_container_width=True
    )

    csv_data = filtered_risk.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Risk Analysis CSV",
        data=csv_data,
        file_name="transaction_risk_analysis.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 4 - TRANSACTION PREDICTION
# ============================================================

elif page == "Transaction Prediction":

    st.title("🔍 Transaction Fraud Prediction")

    st.write(
        "Enter transaction details to estimate fraud probability "
        "and transaction risk."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        login_attempts = st.number_input(
            "Login Attempts",
            min_value=0,
            value=1,
            step=1
        )

        device_risk_score = st.number_input(
            "Device Risk Score",
            min_value=0.0,
            value=0.5
        )

        transfer_frequency = st.number_input(
            "Transfer Frequency",
            min_value=0,
            value=1,
            step=1
        )

        anomaly_score = st.number_input(
            "Anomaly Score",
            min_value=0.0,
            value=0.5
        )

        account_age_days = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=1000,
            step=1
        )

        transaction_time_hour = st.number_input(
            "Transaction Time (Hour)",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

        failed_transactions_last_30d = st.number_input(
            "Failed Transactions Last 30 Days",
            min_value=0,
            value=0,
            step=1
        )

        avg_monthly_balance = st.number_input(
            "Average Monthly Balance",
            min_value=0.0,
            value=10000.0
        )

        daily_transaction_count = st.number_input(
            "Daily Transaction Count",
            min_value=0,
            value=2,
            step=1
        )

        geo_distance_km = st.number_input(
            "Geo Distance (KM)",
            min_value=0.0,
            value=5.0
        )

    with col2:

        session_duration_minutes = st.number_input(
            "Session Duration (Minutes)",
            min_value=0.0,
            value=10.0
        )

        transaction_velocity_score = st.number_input(
            "Transaction Velocity Score",
            min_value=0.0,
            value=1.0
        )

        payment_channel = st.selectbox(
            "Payment Channel",
            sorted(df["payment_channel"].unique())
        )

        authentication_type = st.selectbox(
            "Authentication Type",
            sorted(df["authentication_type"].unique())
        )

        card_present_flag = st.selectbox(
            "Card Present",
            [0, 1]
        )

        international_transaction_flag = st.selectbox(
            "International Transaction",
            [0, 1]
        )

        suspicious_ip_flag = st.selectbox(
            "Suspicious IP",
            [0, 1]
        )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Transaction Risk",
        type="primary"
    )

    if predict_button:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame([{
            "transaction_amount": transaction_amount,
            "login_attempts": login_attempts,
            "device_risk_score": device_risk_score,
            "transfer_frequency": transfer_frequency,
            "anomaly_score": anomaly_score,
            "account_age_days": account_age_days,
            "transaction_time_hour": transaction_time_hour,
            "failed_transactions_last_30d":
                failed_transactions_last_30d,
            "avg_monthly_balance": avg_monthly_balance,
            "daily_transaction_count":
                daily_transaction_count,
            "geo_distance_km": geo_distance_km,
            "session_duration_minutes":
                session_duration_minutes,
            "transaction_velocity_score":
                transaction_velocity_score,
            "card_present_flag": card_present_flag,
            "international_transaction_flag":
                international_transaction_flag,
            "suspicious_ip_flag": suspicious_ip_flag
        }])

        # ----------------------------------------------------
        # Feature Engineering
        # ----------------------------------------------------

        input_data["behavior_risk_score"] = (
            input_data["login_attempts"] * 2
            + input_data["failed_transactions_last_30d"] * 2
            + input_data["suspicious_ip_flag"] * 5
            + input_data["international_transaction_flag"] * 2
        )

        input_data["transaction_activity_score"] = (
            input_data["daily_transaction_count"]
            + input_data["transfer_frequency"]
            + input_data["transaction_velocity_score"]
        )

        input_data["new_account_flag"] = (
            input_data["account_age_days"] < 365
        ).astype(int)

        amount_threshold = df[
            "transaction_amount"
        ].quantile(0.75)

        input_data["high_transaction_amount_flag"] = (
            input_data["transaction_amount"]
            > amount_threshold
        ).astype(int)

        # ----------------------------------------------------
        # One-Hot Encoding
        # ----------------------------------------------------

        payment_column = (
            "payment_channel_" + payment_channel
        )

        authentication_column = (
            "authentication_type_" + authentication_type
        )

        input_data[payment_column] = 1
        input_data[authentication_column] = 1

        # ----------------------------------------------------
        # Match model feature order
        # ----------------------------------------------------

        for column in feature_names:

            if column not in input_data.columns:
                input_data[column] = 0

        input_data = input_data[
            feature_names
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        fraud_probability = model.predict_proba(
            input_data
        )[0][1]

        risk_score = fraud_probability * 100

        if risk_score < 25:
            risk_level = "Low Risk"
        elif risk_score < 50:
            risk_level = "Medium Risk"
        elif risk_score < 75:
            risk_level = "High Risk"
        else:
            risk_level = "Critical Risk"

        prediction = model.predict(
            input_data
        )[0]

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

        col2.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )

        col3.metric(
            "Risk Level",
            risk_level
        )

        if prediction == 1:

            st.error(
                "⚠️ The model classified this transaction "
                "as potentially fraudulent."
            )

        else:

            st.success(
                "✅ The model classified this transaction "
                "as non-fraudulent."
            )

        st.info(
            "Risk thresholds used in this project are "
            "project-defined thresholds and are not official "
            "banking regulatory standards."
        )


# ============================================================
# PAGE 5 - MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Comparison of machine learning models used for "
        "fraud detection."
    )

    st.markdown("---")

    st.subheader("Model Comparison")

    st.dataframe(
        model_comparison,
        use_container_width=True
    )

    st.markdown("---")

    # F1 Score comparison
    if "F1" in model_comparison.columns:

        st.subheader("F1 Score Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            model_comparison["Model"],
            model_comparison["F1"]
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("F1 Score")
        ax.set_title("Model F1 Score Comparison")

        plt.xticks(rotation=20)

        st.pyplot(fig)

    # ROC-AUC comparison
    if "ROC-AUC" in model_comparison.columns:

        st.subheader("ROC-AUC Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            model_comparison["Model"],
            model_comparison["ROC-AUC"]
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Model ROC-AUC Comparison")

        plt.xticks(rotation=20)

        st.pyplot(fig)

    st.markdown("---")

    st.success(
        "Random Forest was selected as the final model "
        "because it achieved the highest F1 score among "
        "the evaluated models."
    )

    st.info(
        "The model was trained using a stratified train-test "
        "split with random_state=42."
    )