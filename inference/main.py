#!/usr/bin/env python3
"""
Run FULL experiment pipeline from experiment.yaml

Pipeline:
---------
1) OOF citywide inference (oof_predictions.csv)
2) OOF metrics computation (oof_metrics.csv)
3) OOF confusion matrix (PNG)
4) Full-resolution rasterization

All paths and settings are inferred from experiment.yaml
"""

import os
import yaml
import argparse

from inference.run_city_inference import run_city_inference
from inference.compute_oof_metrics import run_oof_metrics
from inference.run_rasterize_fullres import run_rasterize_fullres
from inference.compute_confusion_matrix import run_confusion_matrix


# ============================================================
# UTILITIES
# ============================================================
def build_oof_path(cfg):
    model_cfg = cfg["model"]
    fusion = model_cfg["fusion_type"]
    sensor = model_cfg.get("sensor_type", "fusion")
    prisma_flag = "prisma" if model_cfg.get("use_prisma", False) else "no_prisma"

    parts = [cfg["inference_save_dir"], fusion]
    if fusion == "single":
        parts.append(sensor)
    parts.append(prisma_flag)

    return os.path.join(*parts, "oof_predictions.csv")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_full_experiment(cfg_path, valid_mask_tif=None, force=False):

    print("\n================ FULL EXPERIMENT PIPELINE ================\n")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------------
    # 1) OOF CITYWIDE INFERENCE
    # --------------------------------------------------------
    oof_csv = build_oof_path(cfg)

    if not os.path.exists(oof_csv) or force:
        print("\n▶ STEP 1: Running OOF citywide inference\n")
        run_city_inference(
            cfg_path=cfg_path,
            valid_mask_tif=valid_mask_tif
        )
    else:
        print(f"\n✔ STEP 1 skipped (OOF predictions already exist):\n  {oof_csv}")

    # --------------------------------------------------------
    # 2) OOF METRICS
    # --------------------------------------------------------
    print("\n▶ STEP 2: Computing OOF metrics\n")
    oof_dir = os.path.dirname(oof_csv)
    run_oof_metrics(oof_csv=oof_csv, out_dir=oof_dir)
    print(f"✔ OOF metrics saved to:\n  {oof_dir}")

    # --------------------------------------------------------
    # 3) OOF CONFUSION MATRIX
    # --------------------------------------------------------
    print("\n▶ STEP 3: Plotting OOF confusion matrix\n")
    run_confusion_matrix(cfg_path=cfg_path)

    # --------------------------------------------------------
    # 4) FULL-RESOLUTION RASTERIZATION
    # --------------------------------------------------------
    if os.path.exists(oof_csv):
        print("\n▶ STEP 4: Building full-resolution probability map\n")
        run_rasterize_fullres(cfg_path)
    else:
        print("\n⚠ STEP 4 skipped (OOF CSV not found)")

    print("\n================ PIPELINE COMPLETED ======================\n")

# ============================================================
# CLI
# ============================================================
# Example usage:
# --------------
# Standard run using default experiment.yaml:
#   python -m inference.main
#   # Metrics will be written to the same inference folder as predictions (oof_metrics.csv)
#   # Also generates confusion_matrix_oof.png in the inference folder
#
# With explicit config file and SAR valid mask:
#   python -m inference.main \
#       --cfg training/configs/experiment.yaml \
#       --valid_mask_tif path/to/sar_valid_mask.tif
#
# Force re-run of inference even if OOF predictions already exist:
#   python -m inference.main --force

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run full slum-mapping experiment from experiment.yaml"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="training/configs/experiment.yaml",
        help="Path to experiment YAML config"
    )

    parser.add_argument(
        "--valid_mask_tif",
        type=str,
        default=None,
        help="Optional SAR valid-mask GeoTIFF"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of inference even if OOF CSV exists"
    )

    args = parser.parse_args()

    run_full_experiment(
        cfg_path=args.cfg,
        valid_mask_tif=args.valid_mask_tif,
        force=args.force
    )