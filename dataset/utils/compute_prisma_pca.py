import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from tqdm import tqdm

def compute_prisma_pca_per_fold(
    metadata_csv,
    folds_csv,
    out_dir,
    n_components=20
):
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    folds = pd.read_csv(folds_csv)[["patch_id", "fold"]]

    df = meta.merge(folds, on="patch_id", how="inner")

    for fold in sorted(df["fold"].unique()):
        print(f"\n▶ Computing PCA for fold {fold}")

        # -----------------------------
        # TRAIN = all folds except f
        # -----------------------------
        train_df = df[df["fold"] != fold]

        X = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            cube = np.load(row["prisma_path"])["data"]  # [230, H, W]
            spec = cube.mean(axis=(1, 2))                # [230]
            X.append(spec)

        X = np.stack(X)  # [N, 230]

        print(f"  → Training samples: {X.shape[0]}")
        print(f"  → Input dim: {X.shape[1]}")

        pca = PCA(
            n_components=n_components,
            whiten=True,
            random_state=0
        )
        pca.fit(X)

        out_path = os.path.join(out_dir, f"prisma_pca_fold{fold}.joblib")
        joblib.dump(pca, out_path)

        explained = pca.explained_variance_ratio_.sum()
        print(f"  ✔ Saved PCA → {out_path}")
        print(f"  ✔ Explained variance: {explained:.3f}")

if __name__ == "__main__":
    compute_prisma_pca_per_fold(
        metadata_csv="dataset/metadata.csv",
        folds_csv="dataset/folds.csv",
        out_dir="dataset/prisma_pca_weights",
        n_components=16
    )