import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import argparse

"""
PRISMA PCA OFFLINE PREPROCESSING (OUT-OF-FOLD SAFE)
==================================================

This script performs an **offline PCA projection of PRISMA hyperspectral patches**
in order to remove the PCA computation from the Dataset __getitem__ and
significantly speed up training and inference.

--------------------------------------------------
WHEN THIS SCRIPT MUST BE RUN
--------------------------------------------------
This script is intended to be executed **after**:

1. The PRISMA patches have already been extracted and saved as `.npz` files
   (e.g. C = 230 hyperspectral bands, H x W spatial size), typically in:
       dataset/patches_prisma/

2. The PCA models have already been trained **per fold** using an
   out-of-fold strategy and saved as:
       prisma_pca_fold{fold}.joblib

   where each PCA model is trained **only on the training folds**
   corresponding to that fold, to avoid data leakage.

--------------------------------------------------
WHAT THIS SCRIPT DOES
--------------------------------------------------
• Reads the metadata and folds CSV files to associate each patch with its fold  
• For each fold:
    - loads the corresponding PCA model (prisma_pca_fold{k}.joblib)
    - applies that PCA to all PRISMA patches belonging to that fold
• Saves the PCA-reduced PRISMA patches:
    - with the SAME patch filename
    - in a NEW output directory (e.g. dataset/patches_prisma_pca/)
    - preserving spatial layout (H x W)
    - with reduced spectral dimension (C = n_components)

The original PRISMA patches are NEVER modified.

--------------------------------------------------
WHY THIS IS IMPORTANT
--------------------------------------------------
• Keeps the entire pipeline **out-of-fold safe**
• Removes heavy PCA computations from Dataset.__getitem__
• Dramatically speeds up training and inference
• Makes the Dataset CPU-light and GPU-friendly
• Ensures reproducibility and clean experimental design

--------------------------------------------------
EXPECTED INPUT / OUTPUT
--------------------------------------------------
Input:
    dataset/patches_prisma/patch_xxxxxx.npz
        → shape: [C_original, H, W]

Output:
    dataset/patches_prisma_pca/patch_xxxxxx.npz
        → shape: [n_components, H, W]

After running this script, the Dataset should directly load
the PCA-reduced PRISMA patches without any on-the-fly transformation.
"""

def main(args):
    os.makedirs(args.prisma_out_dir, exist_ok=True)

    # -------------------------------------------------
    # LOAD METADATA + FOLDS
    # -------------------------------------------------
    meta = pd.read_csv(args.metadata_csv)
    folds = pd.read_csv(args.folds_csv)

    df = meta.merge(
        folds[["patch_id", "fold"]],
        on="patch_id",
        how="inner"
    )

    if df.empty:
        raise RuntimeError("No patches found after merge!")

    print(f"✔ Loaded {len(df)} patches")

    # -------------------------------------------------
    # PROCESS FOLD BY FOLD (OOF SAFE)
    # -------------------------------------------------
    for fold in sorted(df["fold"].unique()):
        print(f"\n========== FOLD {fold} ==========")

        df_fold = df[df["fold"] == fold]

        pca_path = os.path.join(
            args.pca_weights_dir,
            f"prisma_pca_fold{fold}.joblib"
        )

        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing PCA: {pca_path}")

        print(f"→ Loading PCA: {pca_path}")
        pca = joblib.load(pca_path)

        for _, row in tqdm(df_fold.iterrows(), total=len(df_fold)):
            patch_id = row["patch_id"]
            in_path = row["prisma_path"]

            out_path = os.path.join(
                args.prisma_out_dir,
                os.path.basename(in_path)
            )

            if os.path.exists(out_path):
                continue  # already processed

            # -----------------------------
            # LOAD PRISMA PATCH
            # -----------------------------
            obj = np.load(in_path)
            prisma = obj["data"].astype("float32")   # [C, H, W]

            C, H, W = prisma.shape

            # -----------------------------
            # APPLY PCA
            # -----------------------------
            prisma_flat = prisma.reshape(C, -1).T    # [H*W, C]
            z = pca.transform(prisma_flat)           # [H*W, n_components]
            z = z.T.reshape(-1, H, W)                # [n_components, H, W]

            # -----------------------------
            # SAVE
            # -----------------------------
            np.savez_compressed(
                out_path,
                data=z
            )

        print(f"Fold {fold} completed")

    print("\nAll PRISMA patches preprocessed with PCA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline PCA projection of PRISMA hyperspectral patches using precomputed weights")
    parser.add_argument("--metadata_csv", type=str, default="dataset/metadata.csv", help="Path to metadata CSV file")
    parser.add_argument("--folds_csv", type=str, default="dataset/folds.csv", help="Path to folds CSV file")
    parser.add_argument("--prisma_in_dir", type=str, default="dataset/patches_prisma", help="Input directory for PRISMA patches")
    parser.add_argument("--prisma_out_dir", type=str, default="dataset/patches_prisma_pca", help="Output directory for PCA-reduced PRISMA patches")
    parser.add_argument("--pca_weights_dir", type=str, default="dataset/prisma_pca_weights", help="Directory containing PCA model weights per fold")
    parser.add_argument("--n_components", type=int, default=16, help="Number of PCA components")

    args = parser.parse_args()
    main(args)