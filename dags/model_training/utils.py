from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_performance_plots(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
) -> Dict[str, any]:
    """
    Get performance plots.

    parameters:
    ------
    y_true: (pd.DataFrame)
        True labels.

    y_pred: (pd.DataFrame)
        Predicted labels.

    prefix: (str)
        Prefix for the plot names.

    return:
    ------
        Performance plots.
    """
    # for roc curve
    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    # for confusion matrix
    cm_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    # for precision recall curve
    pr_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": cm_figure,
        f"{prefix}_precision_recall_curve": pr_figure,
    }


def get_classification_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
) -> Dict[str, float]:
    """
    Log classification metrics.

    Parameters:
    ------
    y_true: (pd.DataFrame)
        True labels.

    y_pred: (pd.DataFrame)
        Predicted labels.

    prefix: (pd.DataFrame)
        Prefix for the metric names.

    return:
    ------
        Classification metrics.
    """
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred),
    }

    return metrics
