import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob):

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_prob = torch.cat(y_prob).cpu().numpy()

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    metrics["precision"] = precision_score(
        y_true, y_pred, zero_division=0
    )

    metrics["recall"] = recall_score(
        y_true, y_pred, zero_division=0
    )

    metrics["f1"] = f1_score(
        y_true, y_pred, zero_division=0
    )

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except:
        metrics["auc"] = 0.0

    return metrics