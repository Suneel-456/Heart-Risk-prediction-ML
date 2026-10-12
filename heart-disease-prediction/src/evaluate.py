"""Metrics, plots and SHAP explainability."""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

log = logging.getLogger(__name__)


def compute_metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def save_plots(y_true, proba, threshold, out=config.REPORTS_DIR):
    out.mkdir(parents=True, exist_ok=True)
    pred = (proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Precision-Recall curve (test set)")
    fig.tight_layout(); fig.savefig(out / "pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)


def save_shap(pipeline, X_train, X_test, out=config.REPORTS_DIR):
    """Global feature importance with SHAP, computed on the transformed features."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        pre = pipeline[:-1]
        clf = pipeline[-1]
        Xt_train = pre.transform(X_train)
        Xt_test = pre.transform(X_test)
        names = pre[-1].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        Xt_train = pd.DataFrame(np.asarray(Xt_train), columns=names)
        Xt_test = pd.DataFrame(np.asarray(Xt_test), columns=names)
        explainer = shap.Explainer(clf, Xt_train)
        sv = explainer(Xt_test)
        if sv.values.ndim == 3:  # (samples, features, classes) -> positive class
            sv = sv[..., 1]
        plt.figure()
        shap.summary_plot(sv, Xt_test, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        imp = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
        imp.rename("mean_abs_shap").to_csv(out / "feature_importance.csv")
    except Exception as exc:  # explainability must never break training
        log.warning("SHAP step skipped: %s", exc)
