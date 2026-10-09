# Banking Fraud Detection & Risk Analytics

An end-to-end machine learning project that analyzes banking transactions, predicts potential fraud, evaluates classification models, and assigns transaction risk levels through an interactive Streamlit dashboard.

**Author:** M Madhan Mohan Reddy

**GitHub Repository:** [View Source Code](https://github.com/madhanmohanm83/banking-fraud-detection-risk-analytics)

**Live Dashboard:** [Launch the Banking Fraud Detection Dashboard](https://banking-fraud-detection-risk-analytics-6oxtrupgnpfggmpzqrlzrk.streamlit.app/)


---

## 1. Project Overview

Banking fraud can lead to financial losses and reduced customer trust. This project explores transaction data and uses machine learning to identify transactions that may be fraudulent.

The project includes data analysis, preprocessing, feature engineering, model training, model evaluation, transaction risk scoring, and an interactive dashboard.

This is an educational machine learning project. Its predictions are not a substitute for a bank's production fraud monitoring system.

## 2. Project Objectives

* Analyze banking transaction patterns.
* Perform exploratory data analysis (EDA).
* Clean and preprocess transaction data.
* Create additional fraud-related features.
* Train and compare multiple classification models.
* Evaluate model performance using standard metrics.
* Estimate fraud probabilities for test transactions.
* Categorize transactions by risk level.
* Present results through a Streamlit dashboard.

## 3. Technologies Used

| Technology     | Purpose                              |
| -------------- | ------------------------------------ |
| Python         | Main programming language            |
| Pandas         | Data loading and manipulation        |
| NumPy          | Numerical operations                 |
| Scikit-learn   | Machine learning and evaluation      |
| Matplotlib     | Data visualization                   |
| Seaborn        | Statistical visualizations           |
| Joblib         | Saving and loading the trained model |
| Streamlit      | Interactive web dashboard            |
| Git and GitHub | Version control and project hosting  |

## 4. Dataset

The project uses the Banking Fraud Detection & Risk Analytics dataset available on Kaggle.

**Dataset source:** [Banking Fraud Detection & Risk Analytics — Kaggle](https://www.kaggle.com/datasets/deepeshkansoti/banking-fraud-detection-and-risk-analyticsdataset)

* Original dataset size: 10,000 transactions
* Original columns: 20
* Target column: `fraud_flag`
* Target classes: fraudulent and non-fraudulent transactions

Important transaction features include transaction amount, login attempts, device risk score, transfer frequency, anomaly score, account age, failed transactions, geographical distance, transaction velocity, payment channel, authentication type, and suspicious IP indicators.

The raw CSV is not included in this repository. Download it from Kaggle and place it at `data/banking_transactions.csv` before running the complete pipeline locally.

## 5. Project Workflow

```text
Banking Transaction Dataset
            |
            v
Exploratory Data Analysis
            |
            v
Data Preprocessing
            |
            v
Feature Engineering
            |
            v
Train-Test Split
            |
            v
Model Training
            |
            v
Model Evaluation and Comparison
            |
            v
Selected Random Forest Model
            |
            v
Fraud Probability and Risk Scoring
            |
            v
Streamlit Dashboard
```

### Stage 1: Exploratory Data Analysis

The analysis examines transaction distributions, fraud-class distribution, relationships between transaction amounts and fraud labels, device risk scores, anomaly scores, and feature correlations.

EDA charts are saved in `results/graphs/`.

### Stage 2: Data Preprocessing

The preprocessing pipeline removes duplicate records, converts the target to a numeric format, removes the transaction identifier from model features, and encodes categorical columns using one-hot encoding.

The processed dataset contains 10,000 rows and 23 columns.

### Stage 3: Feature Engineering

Additional features are created to represent transaction behavior and activity:

* `behavior_risk_score`
* `transaction_activity_score`
* `new_account_flag`
* `high_transaction_amount_flag`

The data is split into training and testing sets using an 80:20 ratio with stratification.

* Training records: 8,000
* Testing records: 2,000

### Stage 4: Model Training

Three classification models are trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Stage 5: Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC

The Random Forest model is selected based on the project's model-comparison results and is saved as `models/fraud_model.pkl`.

### Stage 6: Transaction Risk Scoring

The trained model estimates a fraud probability for each test transaction. The probability is converted into a risk score between 0 and 100.

The project's configurable risk categories are:

| Risk Score     | Risk Level    |
| -------------- | ------------- |
| Below 25       | Low Risk      |
| 25 to below 50 | Medium Risk   |
| 50 to below 75 | High Risk     |
| 75 to 100      | Critical Risk |

These thresholds are project-defined examples, not official banking risk standards.

### Stage 7: Streamlit Dashboard

The dashboard presents transaction analysis, fraud detection results, model evaluation, and transaction risk information through an interactive interface.

## 6. Model Performance

The saved model comparison reported the following results for the Random Forest model:

| Metric    | Result |
| --------- | -----: |
| Accuracy  |  95.2% |
| Precision |  82.1% |
| Recall    |  78.8% |
| F1 Score  |  80.4% |
| ROC-AUC   |  97.4% |

**Interpretation:**

* Accuracy measures the proportion of correct predictions.
* Precision measures how many transactions predicted as fraud were actually fraud.
* Recall measures how many actual fraudulent transactions were identified.
* F1 score balances precision and recall.
* ROC-AUC measures the model's ability to distinguish between the two classes across classification thresholds.

These metrics describe the evaluation performed on this dataset; they do not guarantee the same performance on real banking transactions.

## 7. Risk Analysis Results

The generated test-set risk report contains 2,000 transactions.

| Risk Level    | Transactions |
| ------------- | -----------: |
| Low Risk      |        1,700 |
| Medium Risk   |           60 |
| High Risk     |          120 |
| Critical Risk |          120 |
| **Total**     |    **2,000** |

Risk categories are based on the configured fraud-probability thresholds. They should be interpreted as model-generated risk indicators, not confirmed fraud cases.

## 8. Project Structure

```text
banking-fraud-detection-risk-analytics/
|
|-- data/
|   |-- processed/
|   |   |-- processed_transactions.csv
|   |-- features/
|       |-- X_train.csv
|       |-- X_test.csv
|       |-- y_train.csv
|       |-- y_test.csv
|
|-- models/
|   |-- fraud_model.pkl
|   |-- feature_names.csv
|
|-- results/
|   |-- graphs/
|   |-- model_evaluation/
|   |-- risk_analysis/
|
|-- src/
|   |-- data_analysis.py
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   |-- risk_scoring.py
|
|-- app.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## 9. How to Run Locally

### Prerequisites

* Python installed
* Git installed
* The Kaggle CSV downloaded into `data/banking_transactions.csv`

### Installation

Clone the repository:

```bash
git clone https://github.com/madhanmohanm83/banking-fraud-detection-risk-analytics.git
cd banking-fraud-detection-risk-analytics
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the project pipeline

Execute the scripts from the project root in this order:

```bash
python src/data_analysis.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_training.py
python src/risk_scoring.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal, usually `http://localhost:8501`.

## 10. Results and Visualizations

The repository includes generated outputs:

* Fraud distribution chart
* Transaction amount distribution
* Fraud versus transaction amount
* Correlation heatmap
* Device risk score comparison
* Anomaly score comparison
* Model confusion matrices
* Model comparison metrics
* Transaction risk score report

Browse the `results/` directory to inspect these outputs.

## 11. Limitations and Future Improvements

Potential improvements include:

* Testing on additional datasets and unseen transaction patterns.
* Tuning classification thresholds to balance missed fraud and false alerts.
* Adding cross-validation and hyperparameter optimization.
* Monitoring model drift and prediction quality.
* Adding downloadable reports and interactive transaction filters.
* Deploying the dashboard publicly with appropriate data and security controls.

## 12. Disclaimer

This project is intended for learning, demonstration, and portfolio purposes. It uses a Kaggle dataset and model-defined risk thresholds. Predictions should not be treated as verified fraud findings or used as the sole basis for financial decisions.

## Author

**M Madhan Mohan Reddy**

[GitHub Profile](https://github.com/madhanmohanm83)

[Project Repository](https://github.com/madhanmohanm83/banking-fraud-detection-risk-analytics)
