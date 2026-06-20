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

    # Autolog params/metrics but log the model ourselves with an explicit conda
    # env so the image built by `mlflow models build-docker --env-manager conda`
    # uses conda-forge + nodefaults (avoids the Anaconda defaults-channel ToS
    # gate that breaks the build on CI runners).
    mlflow.sklearn.autolog(log_models=False)

    conda_env = {
        "name": "telco_churn_env",
        "channels": ["conda-forge", "nodefaults"],
        "dependencies": [
            "python=3.12.7",
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "scikit-learn==1.5.2",
                    "pandas==2.3.3",
                    "numpy==2.4.6",
                    "cloudpickle==3.1.2",
                ]
            },
        ],
    }

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

        # Log the model with the explicit conda env (conda-forge/nodefaults).
        mlflow.sklearn.log_model(model, artifact_path="model", conda_env=conda_env)

        # Persist the run id so the next CI step can locate the logged model.
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID: {run.info.run_id}")
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
