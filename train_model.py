"""Train and evaluate the fraud classifier; SMOTE is applied to training data only."""
from pathlib import Path
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "transactions_last_24h",
    "merchant_risk_score",
]


def main():
    data_path = ROOT / "data" / "transactions.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run python generate_data.py first.")

    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:\n",
          classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "fraud_detection_pipeline.joblib"

    joblib.dump(
        {"pipeline": model, "features": FEATURES},
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
