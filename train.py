"""Train and evaluate credit risk classifiers on the UCI German Credit dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "risk",
]
NUMERIC = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "existing_credits", "dependents",
    "amount_per_month",
]
EXCLUDED = ["personal_status_sex", "age", "foreign_worker"]


def main():
    data = pd.read_csv(ROOT / "data/german.data", sep=r"\s+", header=None, names=COLUMNS)
    y = data.pop("risk").map({1: 0, 2: 1})
    if y.isna().any() or data.isna().any().any() or data["duration_months"].le(0).any():
        raise ValueError("Unexpected credit data. Use the included UCI dataset.")
    features = data.drop(columns=EXCLUDED).assign(
        amount_per_month=data["credit_amount"] / data["duration_months"]
    )
    categorical = features.columns.difference(NUMERIC).tolist()
    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, stratify=y, random_state=SEED
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    rows = []
    for name, estimator in models.items():
        preprocess = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        model = Pipeline([("preprocess", preprocess), ("classifier", estimator)])
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]

        for threshold in (0.5, 1 / 6):
            predicted = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predicted, average="binary", zero_division=0
            )
            rows.append({
                "model": name, "threshold": threshold,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision, "recall": recall, "f1": f1,
                "roc_auc": roc_auc_score(y_test, scores),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "uci_cost": int(fp + 5 * fn),
            })

    results = pd.DataFrame(rows)
    print(f"Training rows: {len(y_train)} | Test rows: {len(y_test)}")
    print(results.round(3).to_string(index=False))
    results.to_csv(ROOT / "results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
