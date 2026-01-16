#!/usr/bin/env python3
"""
Out-of-Fold (OOF) Citywide Inference Module
------------------------------------------

Produces citywide slum probability predictions using trained OOF models.
Optionally enforces a SAR-based valid-mask to suppress predictions over
no-data regions (e.g. oblique COSMO-SkyMed borders).

If a patch intersects even a single invalid SAR pixel, its probability is set to 0.
"""

import os
import yaml
import torch
import rasterio
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset.dataset import SlumDataset
from training.models.utils import load_backbone_config
from training.models.model_factory import ModelConfig, build_model_from_cfg


# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# MAIN INFERENCE
# ============================================================
def run_city_inference(cfg_path, valid_mask_tif=None):

    # ------------------------
    # Load config
    # ------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg_dict = cfg["model"]
    num_folds = cfg["num_folds"]

    mode = data_cfg.get("mode", "fusion")
    model_cfg_dict["mode"] = mode
    model_cfg_dict["sar_channels"] = data_cfg.get("sar_channels")
    model_cfg_dict["planet_channels"] = data_cfg.get("planet_channels")

    if mode != "fusion":
        model_cfg_dict["fusion_type"] = "none"

    device = get_device()
    print(f"→ Using device: {device}")

    backbones_cfg = load_backbone_config()

    # ------------------------
    # Load VALID MASK (optional)
    # ------------------------
    valid_mask = None
    if valid_mask_tif is not None:
        print(f"→ Loading valid mask from: {valid_mask_tif}")
        with rasterio.open(valid_mask_tif) as src:
            sar_ref = src.read(1)
            nodata = (src == 0)

        if nodata is None:
            raise RuntimeError("SAR valid-mask GeoTIFF has no nodata value")

        valid_mask = sar_ref != nodata
        print("✔ Valid mask loaded")

    # ------------------------
    # Output paths
    # ------------------------
    infer_root = os.path.join(
        cfg["inference_save_dir"],
        mode,
        model_cfg_dict["fusion_type"],
        model_cfg_dict["backbone_sar"]
    )
    os.makedirs(infer_root, exist_ok=True)
    out_csv = os.path.join(infer_root, "oof_predictions.csv")

    folds_df = pd.read_csv(data_cfg["folds_csv"])
    all_rows = []

    # ============================================================
    # LOOP OVER FOLDS
    # ============================================================
    for fold in range(1, num_folds + 1):

        print(f"\n========== INFERENCE FOLD {fold} ==========")

        df_fold = folds_df[folds_df["fold"] == fold]
        if df_fold.empty:
            print("⚠ No patches in this fold, skipping")
            continue

        ds = SlumDataset(
            metadata_csv=data_cfg["metadata_csv"],
            folds_csv=data_cfg["folds_csv"],
            folds_to_use=[fold],
            mode=mode,
            normalize=data_cfg["normalize"],
            augment=False,
            return_id=True
        )

        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)

        # ------------------------
        # Build model
        # ------------------------
        model_cfg = ModelConfig(**model_cfg_dict)
        model = build_model_from_cfg(backbones_cfg, model_cfg)
        model.to(device)
        model.eval()

        ckpt_path = os.path.join(
            cfg["save_dir"],
            mode,
            model_cfg_dict["fusion_type"],
            model_cfg_dict["backbone_sar"],
            f"fold_{fold}",
            "weights.pt"
        )

        if not os.path.exists(ckpt_path):
            print("⚠ Missing checkpoint, skipping fold")
            continue

        print(f"→ Loading weights: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # ========================================================
        # INFERENCE
        # ========================================================
        with torch.no_grad():
            for inputs, labels, patch_ids in tqdm(
                dl, desc=f"Inference fold {fold}", leave=False
            ):

                labels = labels.numpy().astype(int)
                probs = np.zeros(len(patch_ids), dtype=np.float32)

                for i, pid in enumerate(patch_ids):

                    # ------------------------------------------
                    # VALID MASK CHECK (patch-level)
                    # ------------------------------------------
                    if valid_mask is not None:
                        row = ds.df.loc[ds.df["patch_id"] == pid].iloc[0]

                        x0, y0 = int(row["x"]), int(row["y"])
                        x1 = x0 + cfg["patch"]["size"]
                        y1 = y0 + cfg["patch"]["size"]

                        if not valid_mask[y0:y1, x0:x1].all():
                            probs[i] = 0.0
                            continue

                    # ------------------------------------------
                    # MODEL FORWARD
                    # ------------------------------------------
                    if mode == "fusion":
                        sar, planet = inputs
                        logits = model(
                            sar[i:i+1].to(device),
                            planet[i:i+1].to(device)
                        )
                    elif mode == "sar":
                        logits = model(inputs[i:i+1].to(device))
                    elif mode == "planet":
                        logits = model(inputs[i:i+1].to(device))
                    else:
                        raise ValueError("Invalid mode")

                    probs[i] = torch.sigmoid(
                        logits.squeeze(1)
                    ).cpu().item()

                for pid, p, lab in zip(patch_ids, probs, labels):
                    all_rows.append({
                        "patch_id": pid,
                        "fold": fold,
                        "prob_slum": float(p),
                        "label": int(lab),
                    })

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n✔ Saved OOF predictions to: {out_csv}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OOF citywide inference with optional SAR valid-mask filtering"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="training/configs/experiment.yaml",
        help="Experiment YAML config"
    )

    parser.add_argument(
        "--valid_mask_tif",
        type=str,
        default='/Users/luigi/Desktop/PHD/PROGETTI_E_ATTIVITA_PHD/HEATCORDOBA/slums/Python_Project_HS_MS/preprocessing/coregistered_outputs_3m/CSK2_L1D_20250728_3m.tif',
        help="SAR GeoTIFF used to compute valid mask (nodata-aware)"
    )

    args = parser.parse_args()

    print("\n▶ Starting citywide OOF inference\n")
    run_city_inference(
        cfg_path=args.cfg,
        valid_mask_tif=args.valid_mask_tif
    )
    print("\n✔ Citywide inference completed\n")