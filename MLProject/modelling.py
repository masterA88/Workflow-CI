"""MLflow Project entry point: train the Telco Churn model for CI.

Designed to run via ``mlflow run`` inside GitHub Actions. It logs to a LOCAL
file-store (./mlruns) so the run + model artifact can be uploaded by the CI job
and packaged into a Docker image with ``mlflow models build-docker``.

Author: Hilmi (https://master-hilmi.vercel.app/)
"""

from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_preprocessing")


def load_split(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(DATA_DIR)

    mlflow.sklearn.autolog()

    # When executed via `mlflow run`, MLflow has already created the run and
    # exposes it through MLFLOW_RUN_ID; calling start_run() then resumes it.
    # Only set the experiment when running standalone (python modelling.py).
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("telco_churn_ci")

    with mlflow.start_run(run_name="ci_random_forest") as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))

        # Persist the run id so the next CI step can locate the logged model.
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID: {run.info.run_id}")
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
