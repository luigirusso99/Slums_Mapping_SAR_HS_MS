
import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def run_confusion_matrix(cfg_path: str) -> str:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    infer_root = cfg["inference_save_dir"]
    model_cfg = cfg["model"]
    fusion = model_cfg["fusion_type"]
    use_prisma = model_cfg.get("use_prisma", False)
    prisma_folder = "prisma" if use_prisma else "no_prisma"

    if fusion == "single":
        sensor = model_cfg["sensor_type"]
        oof_dir = os.path.join(
            infer_root,
            fusion,
            sensor,
            prisma_folder,
        )
    else:
        oof_dir = os.path.join(
            infer_root,
            fusion,
            prisma_folder,
        )

    oof_csv = os.path.join(oof_dir, "oof_predictions.csv")
    if not os.path.exists(oof_csv):
        raise FileNotFoundError(f"OOF predictions file not found at {oof_csv}")

    df = pd.read_csv(oof_csv)
    y_true = df["label"].values.astype(int)
    y_prob = df["prob_slum"].values.astype(float)

    def best_threshold(labels, probs, n=200):
        thresholds = np.linspace(0, 1, n)
        best_f1, best_t = -1, 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t, best_f1

    t_opt, f1 = best_threshold(y_true, y_prob)
    y_pred = (y_prob >= t_opt).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if fusion == "single":
        sensor = model_cfg["sensor_type"]
        if sensor == "sar":
            modality_label = "Single SAR"
        elif sensor == "planet":
            modality_label = "Single Optical"
        else:
            modality_label = f"Single {sensor.capitalize()}"
    else:
        if fusion == "early":
            modality_label = "Early Fusion"
        elif fusion == "mid":
            modality_label = "Mid Fusion"
        elif fusion == "late":
            modality_label = "Late Fusion"
        else:
            modality_label = "Fusion"

    prisma_label = "PRISMA" if use_prisma else "no PRISMA"
    title = f"ResNet-18\n{modality_label}"
    if prisma_label:
        title += f" ({prisma_label})"
    title += f" (thr={t_opt:.2f})"

    sns.set_theme(style="white", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 14}
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["Non-Slum", "Slum"])
    ax.set_yticklabels(["Non-Slum", "Slum"], rotation=0)

    out_fig = os.path.join(oof_dir, "confusion_matrix_oof.png")
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    return out_fig