"""Evaluation metrics for classification."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }
