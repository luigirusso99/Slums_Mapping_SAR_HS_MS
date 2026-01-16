#!/usr/bin/env python3
"""
Plot OOF Confusion Matrices for all tested backbones.
Threshold is selected by maximizing F1-score under OOF conditions.
Figure style is publication-ready (seaborn).
"""

import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix
)

# ------------------------------------------------------------
# LOAD EXPERIMENT CONFIG
# ------------------------------------------------------------
CFG_PATH = "training/configs/experiment.yaml"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

INFER_ROOT = cfg["inference_save_dir"]
MODE = cfg["data"]["mode"]
FUSION = cfg["model"]["fusion_type"] if MODE == "fusion" else "none"
BACKBONE_ROOT = os.path.join(INFER_ROOT, MODE, FUSION)

# ------------------------------------------------------------
# HUMAN-FRIENDLY NAMES FOR PLOTTING
# ------------------------------------------------------------
NAME_MAP = {
    "resnet18": "ResNet-18",
    # "efficientnet_b0": "EfficientNet-B0",
    # "mobilenet_v3_small": "MobileNetV3-Small",
    # "convnext_tiny": "ConvNeXt-Tiny",
    # "swin_tiny_patch4_window7_224": "Swin Transformer Tiny"
}

# ------------------------------------------------------------
# DISCOVER BACKBONES AUTOMATICALLY
# ------------------------------------------------------------
BACKBONES = sorted([
    d for d in os.listdir(BACKBONE_ROOT)
    if os.path.isdir(os.path.join(BACKBONE_ROOT, d))
])

print(f"Found backbones: {BACKBONES}")

# ------------------------------------------------------------
# THRESHOLD SELECTION (OOF, max F1)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# COMPUTE CONFUSION MATRICES
# ------------------------------------------------------------
cms = {}
thresholds = {}

for backbone in BACKBONES:
    oof_csv = os.path.join(BACKBONE_ROOT, backbone, "oof_predictions.csv")

    if not os.path.exists(oof_csv):
        print(f"[WARN] Missing OOF for {backbone}")
        continue

    df = pd.read_csv(oof_csv)
    y_true = df["label"].values.astype(int)
    y_prob = df["prob_slum"].values.astype(float)

    t_opt, f1 = best_threshold(y_true, y_prob)
    y_pred = (y_prob >= t_opt).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    cms[backbone] = cm
    thresholds[backbone] = t_opt

# ------------------------------------------------------------
# PLOT GRID OF CONFUSION MATRICES
# ------------------------------------------------------------
sns.set_theme(style="white", font_scale=1.2)

n_models = len(cms)
fig, axes = plt.subplots(
    1, n_models,
    figsize=(4 * n_models, 4),
    constrained_layout=True
)

if n_models == 1:
    axes = [axes]

for ax, (model, cm) in zip(axes, cms.items()):
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

    display_name = NAME_MAP.get(model, model)

    # Determine modality label
    if MODE == "sar":
        modality_label = "Only SAR"
    elif MODE == "planet":
        modality_label = "Only Optical"
    elif MODE == "fusion":
        if FUSION == "early":
            modality_label = "Early Fusion"
        elif FUSION == "mid":
            modality_label = "Middle Fusion"
        elif FUSION == "late":
            modality_label = "Late Fusion"
        else:
            modality_label = "Fusion"
    else:
        modality_label = "Unknown mode"

    ax.set_title(
        f"{display_name} (thr={thresholds[model]:.2f})\n{modality_label}",
        fontsize=13,
        pad=10
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticklabels(["Non-Slum", "Slum"])
    ax.set_yticklabels(["Non-Slum", "Slum"], rotation=0)

# ------------------------------------------------------------
# SAVE FIGURE
# ------------------------------------------------------------
OUT_FIG = os.path.join(BACKBONE_ROOT, "confusion_matrices_oof.png")
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"\n✔ Confusion matrix grid saved to:\n{OUT_FIG}")