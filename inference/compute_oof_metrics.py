# analysis/compute_oof_metrics.py

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score
)

def compute_best_threshold(labels, probs, num_thresholds=200):
    """
    Scans thresholds in [0,1] and finds the one maximizing F1.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = None

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = (precision, recall, f1)

    return best_threshold, best_metrics


def main(oof_csv):
    print("\n▶ Computing OOF metrics from:", oof_csv)

    df = pd.read_csv(oof_csv)

    if not {"prob_slum", "label"}.issubset(df.columns):
        raise ValueError("❌ CSV must contain 'prob_slum' and 'label' columns.")

    probs = df["prob_slum"].values
    labels = df["label"].values.astype(int)

    # -----------------------------
    # GLOBAL METRICS
    # -----------------------------
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    # -----------------------------
    # OPTIMAL THRESHOLD (max F1)
    # -----------------------------
    best_t, (best_prec, best_rec, best_f1) = compute_best_threshold(labels, probs)

    # Metrics at threshold
    preds_opt = (probs >= best_t).astype(int)
    acc_opt = accuracy_score(labels, preds_opt)
    kappa_opt = cohen_kappa_score(labels, preds_opt)

    # -----------------------------
    # PRINT REPORT
    # -----------------------------
    print("\n================ OOF METRICS ================")
    print(f"Total samples: {len(labels)}")
    print("----------------------------------------------")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print("----------------------------------------------")
    print(f"BEST THRESHOLD (max F1):  {best_t:.3f}")
    print("----------------------------------------------")
    print(f"Accuracy:     {acc_opt:.4f}")
    print(f"Cohen's Kappa:{kappa_opt:.4f}")
    print(f"Precision:    {best_prec:.4f}")
    print(f"Recall:       {best_rec:.4f}")
    print(f"F1-score:     {best_f1:.4f}")
    print("==============================================\n")

    print("✔ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        type=str,
        required=True,
        help="Path to oof_predictions.csv"
    )
    args = parser.parse_args()
    main(args.oof)

    ## Example usage:
    # python -m inference.compute_oof_metrics --oof inference/predictions/planet/none/resnet18/oof_predictions.csv