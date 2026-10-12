"""Train, compare and tune models; pick a recall-oriented threshold; log to MLflow.

Run:  python -m src.train
"""
import json
import logging

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import prepare
from src.evaluate import compute_metrics, save_plots, save_shap
from src.feature_engineering import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

RS = config.RANDOM_STATE


def candidates():
    """(estimator, param grid) for each algorithm."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [2, 5]},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=RS),
            {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]},
        ),
        "lightgbm": (
            LGBMClassifier(n_estimators=200, min_child_samples=10, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=RS, verbose=-1),
            {"clf__num_leaves": [4, 8], "clf__learning_rate": [0.03, 0.1]},
        ),
    }


def pick_threshold(y_true, proba, target_recall=config.TARGET_RECALL) -> float:
    """Highest threshold that still reaches the target recall (keeps precision as high as possible)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def main():
    X_train, X_test, y_train, y_test = prepare()
    log.info("Train %s | Test %s | disease rate %.2f", X_train.shape, X_test.shape, y_train.mean())

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    Xf = X_train.astype("float64")  # float schema tolerates missing values at inference
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)

    rows, fitted = [], {}
    for name, (est, grid) in candidates().items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(build_pipeline(est), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            proba = search.best_estimator_.predict_proba(X_test)[:, 1]
            m = compute_metrics(y_test, proba, 0.5)
            m["cv_roc_auc"] = search.best_score_
            mlflow.set_tags({"stage": "model_selection", "algorithm": name})
            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({"cv_folds": config.CV_FOLDS, "train_rows": len(X_train)})
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
            mlflow.sklearn.log_model(
                search.best_estimator_, name="model", serialization_format="cloudpickle",
                signature=infer_signature(Xf, search.best_estimator_.predict_proba(X_train)),
            )
            rows.append({"model": name, **m, "best_params": json.dumps(search.best_params_, default=str)})
            fitted[name] = search.best_estimator_
            log.info("%-20s cv_auc=%.3f test_auc=%.3f recall=%.3f", name, m["cv_roc_auc"], m["roc_auc"], m["recall"])

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best_name = comparison.loc[0, "model"]
    best = fitted[best_name]
    log.info("Best model by CV ROC-AUC: %s", best_name)

    # Choose the threshold on out-of-fold training predictions (no test leakage).
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train, oof)
    test_proba = best.predict_proba(X_test)[:, 1]
    final = compute_metrics(y_test, test_proba, threshold)
    log.info("Threshold %.3f -> test recall %.3f precision %.3f", threshold, final["recall"], final["precision"])

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    bundle = {"pipeline": best, "threshold": threshold, "model_name": best_name,
              "features": config.FEATURES, "metrics": final}
    joblib.dump(bundle, config.MODEL_PATH)

    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    (config.REPORTS_DIR / "final_metrics.json").write_text(
        json.dumps({"model": best_name, **final}, indent=2))
    pd.DataFrame({"y_true": y_test.values, "probability": test_proba.round(4),
                  "prediction": (test_proba >= threshold).astype(int)},
                 index=y_test.index).to_csv(config.REPORTS_DIR / "test_predictions.csv")
    save_plots(y_test, test_proba, threshold)
    save_shap(best, X_train, X_test)

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.set_tags({"stage": "final", "algorithm": best_name})
        mlflow.log_param("model", best_name)
        mlflow.log_param("threshold", round(threshold, 4))
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in final.items()})
        mlflow.log_artifacts(str(config.REPORTS_DIR), artifact_path="reports")
        mlflow.log_artifact(str(config.MODEL_PATH), artifact_path="bundle")
        # Log the model with its signature and register it in the Model Registry.
        info = mlflow.sklearn.log_model(
            best, name="model", serialization_format="cloudpickle",
            signature=infer_signature(Xf, best.predict_proba(X_train)),
            input_example=Xf.head(3),
            registered_model_name=config.MLFLOW_MODEL_NAME,
        )
        log.info("Registered '%s' (%s)", config.MLFLOW_MODEL_NAME, info.model_uri)
    log.info("Saved model to %s", config.MODEL_PATH)
    return comparison, final


if __name__ == "__main__":
    main()
