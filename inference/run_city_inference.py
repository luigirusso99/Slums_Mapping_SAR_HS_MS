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
from training.models.model_factory import build_model


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

    fusion = model_cfg_dict["fusion_type"]
    sensor = model_cfg_dict.get("sensor_type", "fusion")
    use_prisma = model_cfg_dict.get("use_prisma", False)
    prisma_flag = "prisma" if use_prisma else "no_prisma"

    device = get_device()
    print(f"→ Using device: {device}")

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
    infer_parts = [
        cfg["inference_save_dir"],
        fusion,
    ]

    if fusion == "single":
        infer_parts.append(sensor)

    infer_parts.append(prisma_flag)

    infer_root = os.path.join(*infer_parts)
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
            fusion_type=fusion,
            sensor_type=model_cfg_dict.get("sensor_type"),
            normalize=data_cfg["normalize"],
            augment=False,
            return_id=True,
            use_prisma=use_prisma,
            stats_opt_sar_csv=data_cfg["normalization_stats_opt_sar_csv"],
            stats_prisma_csv=data_cfg.get("normalization_stats_prisma_csv"),
        )

        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)

        # ------------------------
        # Build model
        # ------------------------
        model = build_model({
            **model_cfg_dict,
            "sar_channels": data_cfg["sar_channels"],
            "planet_channels": data_cfg["planet_channels"],
            "prisma_channels": data_cfg.get("prisma_channels"),
        })
        model.to(device)
        model.eval()

        ckpt_parts = [
            cfg["save_dir"],
            fusion,
        ]

        if fusion == "single":
            ckpt_parts.append(sensor)

        ckpt_parts.extend([
            prisma_flag,
            f"fold_{fold}",
            "weights.pt",
        ])

        ckpt_path = os.path.join(*ckpt_parts)

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
                    if fusion == "single":
                        if sensor == "sar":
                            if use_prisma:
                                sar, prisma = inputs
                                logits = model(
                                    sar[i:i+1].to(device),
                                    prisma[i:i+1].to(device),
                                )
                            else:
                                logits = model(inputs[i:i+1].to(device))

                        elif sensor == "planet":
                            if use_prisma:
                                planet, prisma = inputs
                                logits = model(
                                    planet[i:i+1].to(device),
                                    prisma[i:i+1].to(device),
                                )
                            else:
                                logits = model(inputs[i:i+1].to(device))

                    else:
                        if use_prisma:
                            sar, planet, prisma = inputs
                            logits = model(
                                sar[i:i+1].to(device),
                                planet[i:i+1].to(device),
                                prisma[i:i+1].to(device),
                            )
                        else:
                            sar, planet = inputs
                            logits = model(
                                sar[i:i+1].to(device),
                                planet[i:i+1].to(device),
                            )

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
        default='/Users/luigi/Desktop/PHD/PROGETTI_E_ATTIVITA_PHD/HEATCORDOBA/slums/PyProj_Slums_Mapping_SAR_MS_HS/raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI_aligned.tif',
        help="SAR GeoTIFF used to compute valid mask (nodata-aware)"
    )

    args = parser.parse_args()

    print("\n▶ Starting citywide OOF inference\n")
    run_city_inference(
        cfg_path=args.cfg,
        valid_mask_tif=args.valid_mask_tif
    )
    print("\n✔ Citywide inference completed\n")