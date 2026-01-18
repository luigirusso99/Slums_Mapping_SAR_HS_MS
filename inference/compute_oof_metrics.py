"""
OOF Metrics Computation Module
------------------------------
Computes evaluation metrics from out-of-fold (OOF) predictions.
Designed to be import-safe and orchestrated by inference/main.py.
"""

import argparse
import os
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


def run_oof_metrics(oof_csv: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

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

    # Write human-readable report
    out_txt = os.path.join(out_dir, "oof_metrics.txt")
    with open(out_txt, "w") as f:
        f.write("OOF METRICS REPORT\n")
        f.write("=================\n")
        f.write(f"Total samples: {len(labels)}\n\n")
        f.write("GLOBAL METRICS\n")
        f.write("--------------\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC:  {pr_auc:.4f}\n\n")
        f.write("OPTIMAL THRESHOLD (max F1)\n")
        f.write("-------------------------\n")
        f.write(f"Threshold: {best_t:.4f}\n\n")
        f.write("METRICS @ OPTIMAL THRESHOLD\n")
        f.write("---------------------------\n")
        f.write(f"Accuracy:  {acc_opt:.4f}\n")
        f.write(f"Kappa:     {kappa_opt:.4f}\n")
        f.write(f"Precision: {best_prec:.4f}\n")
        f.write(f"Recall:    {best_rec:.4f}\n")
        f.write(f"F1-score:  {best_f1:.4f}\n")

    # Save machine-readable CSV
    metrics_df = pd.DataFrame([{
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_t,
        "accuracy": acc_opt,
        "kappa": kappa_opt,
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1,
        "n_samples": len(labels),
    }])
    out_csv = os.path.join(out_dir, "oof_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)

    print(f"✔ OOF metrics saved to: {out_txt}")

    return out_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        type=str,
        required=True,
        help="Path to oof_predictions.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where metrics files will be saved"
    )
    args = parser.parse_args()
    run_oof_metrics(args.oof, args.out_dir)
