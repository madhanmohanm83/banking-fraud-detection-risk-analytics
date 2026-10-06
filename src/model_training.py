import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD TRAINING AND TEST DATA
# ============================================

X_train = pd.read_csv(FEATURE_DIR / "X_train.csv")
X_test = pd.read_csv(FEATURE_DIR / "X_test.csv")

y_train = pd.read_csv(FEATURE_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(FEATURE_DIR / "y_test.csv").squeeze()

print("=" * 60)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("MODEL TRAINING")
print("=" * 60)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# ============================================
# 3. DEFINE MODELS
# ============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

# ============================================
# 4. TRAIN AND EVALUATE MODELS
# ============================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = model

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Fraud", "Fraud"]
    )

    display.plot()

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

# ============================================
# 5. CREATE RESULTS TABLE
# ============================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

# ============================================
# 6. SAVE RESULTS
# ============================================

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

# ============================================
# 7. SELECT MODEL BASED ON F1 SCORE
# ============================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nSelected model based on F1 Score:")
print(best_model_name)

# ============================================
# 8. SAVE BEST MODEL
# ============================================

import joblib

model_path = MODEL_DIR / "fraud_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved successfully!")
print(model_path)

# ============================================
# 9. SAVE FEATURE NAMES
# ============================================

feature_names = pd.DataFrame({
    "feature": X_train.columns
})

feature_names.to_csv(
    MODEL_DIR / "feature_names.csv",
    index=False
)

# ============================================
# 10. FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)