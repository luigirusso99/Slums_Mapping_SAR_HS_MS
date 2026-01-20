# training/dataset.py

import torch, rasterio, pandas as pd, random, numpy as np
from torch.utils.data import Dataset
from training.dataset.utils.augmentations import GeoAugmentations
from training.dataset.utils.normalization import ZScoreNormalizer
import os

class SlumDataset(Dataset):
    """
    Dataset for Slum Mapping.

    PRISMA patches are expected to be preprocessed offline and already
    reduced via PCA. No PCA computation is performed at runtime.
    """

    def __init__(self, metadata_csv, folds_csv, folds_to_use,
                 fusion_type="fusion",
                 sensor_type=None,
                 normalize=True, augment=True, return_id=False,
                 use_prisma=False,
                 stats_opt_sar_csv="dataset/normalization_stats_opt_sar.csv"):
        
        assert fusion_type in ("single", "early", "mid", "late")
        if fusion_type == "single":
            assert sensor_type in ("sar", "planet")

        meta = pd.read_csv(metadata_csv)
        folds = pd.read_csv(folds_csv)

        df = meta.merge(folds[["patch_id", "fold"]], on="patch_id", how="inner")
        df = df[df["fold"].isin(folds_to_use)]

        if df.empty:
            raise RuntimeError("No patches match selected folds!")

        if use_prisma:
            if "prisma_path" not in df.columns:
                raise RuntimeError("prisma_path column missing in metadata but use_prisma=True")

        self.df = df.reset_index(drop=True)

        if fusion_type == "single":
            self.mode = sensor_type
        else:
            self.mode = "fusion"

        self.fusion_type = fusion_type
        self.sensor_type = sensor_type

        self.normalize = normalize
        self.augment = augment
        self.return_id = return_id
        self.use_prisma = use_prisma

        self.fold_ids = sorted(folds_to_use)

        # internal augmentations
        self.aug = GeoAugmentations() if augment else None

        # normalizers are now derived automatically from min/max statistics
        if normalize:
            # Do not build norm_prisma if using PCA (PRISMA raw not supported)
            self.norm_sar, self.norm_planet = self.load_stats_and_build_normalizers_multi(stats_opt_sar_csv)
        else:
            self.norm_sar = None
            self.norm_planet = None

    def load_stats_and_build_normalizers_multi(self, stats_opt_sar_csv):
        """
        Loads per-band statistics from normalization CSVs and builds
        Z-score normalizers for:
        - SAR (single-band backscatter)
        - PlanetScope (8-band multispectral)

        fusion_type="single" corresponds to SAR-only or Planet-only dataset
        fusion_type in ("early","mid","late") corresponds to fusion dataset
        """

        # Load SAR + Planet stats from same CSV (distinguished by sensor column)
        sar_planet_df = pd.read_csv(stats_opt_sar_csv)

        norm_sar = None
        norm_planet = None

        if self.mode in ("sar", "fusion"):
            sar_df = sar_planet_df[sar_planet_df["sensor"] == "sar"].sort_values("band")
            mean_sar = torch.tensor(sar_df["mean"].values, dtype=torch.float32)
            std_sar  = torch.tensor(sar_df["std"].values,  dtype=torch.float32)
            std_sar[std_sar == 0] = 1.0
            norm_sar = ZScoreNormalizer(mean_sar, std_sar)

        if self.mode in ("planet", "fusion"):
            planet_df = sar_planet_df[sar_planet_df["sensor"] == "planet"].sort_values("band")
            mean_planet = torch.tensor(planet_df["mean"].values, dtype=torch.float32)
            std_planet  = torch.tensor(planet_df["std"].values,  dtype=torch.float32)
            std_planet[std_planet == 0] = 1.0
            norm_planet = ZScoreNormalizer(mean_planet, std_planet)

        return norm_sar, norm_planet
    # ------------------------------

    def load(self, path):
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")
        return torch.tensor(arr)
    
    def load_npz(self, path):
        obj = np.load(path)
        arr = obj["data"].astype("float32")   # shape: [B, H, W]
        return torch.from_numpy(arr)

    # ------------------------------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sar = self.load(row["sar_path"]) if self.mode in ("sar", "fusion") else None
        planet = self.load(row["planet_path"]) if self.mode in ("planet", "fusion") else None
        prisma = None
        if self.use_prisma:
            prisma = self.load_npz(row["prisma_path"])

        # Augmentations (SAR / Planet / PRISMA if present)
        if self.aug:
            sar, planet, prisma = self.aug(sar, planet, prisma)

        # Normalization
        if self.normalize:
            if sar is not None and self.norm_sar is not None:
                sar = self.norm_sar(sar)
            if planet is not None and self.norm_planet is not None:
                planet = self.norm_planet(planet)

        y = torch.tensor(row["label"], dtype=torch.long)
        patch_id = row["patch_id"]

        if self.mode == "sar":
            if self.use_prisma:
                sample = (sar, prisma)
            else:
                sample = sar
            if self.return_id:
                return sample, y, patch_id
            else:
                return sample, y

        if self.mode == "planet":
            if self.use_prisma:
                sample = (planet, prisma)
            else:
                sample = planet
            if self.return_id:
                return sample, y, patch_id
            else:
                return sample, y

        if self.mode == "fusion":
            if self.use_prisma:
                sample = (sar, planet, prisma)
            else:
                sample = (sar, planet)
            if self.return_id:
                return sample, y, patch_id
            else:
                return sample, y

    def __len__(self):
        return len(self.df)
    
def main():
    import yaml
    import random
    import torch
    import matplotlib.pyplot as plt
    CFG_PATH = "training/configs/experiment.yaml"
    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    print("\n========== DATASET TEST ==========")
    print(f"Fusion type: {model_cfg['fusion_type']}")
    print(f"Sensor type: {model_cfg.get('sensor_type')}")
    print(f"Use PRISMA:  {model_cfg.get('use_prisma', False)}")

    # --------------------------------------------------
    # Build dataset (single fold for testing)
    # --------------------------------------------------
    dataset = SlumDataset(
        metadata_csv=data_cfg["metadata_csv"],
        folds_csv=data_cfg["folds_csv"],
        folds_to_use=[1, 2, 3, 4],
        fusion_type=model_cfg["fusion_type"],
        sensor_type=model_cfg.get("sensor_type"),
        normalize=data_cfg.get("normalize", True),
        augment=True,
        return_id=True,
        use_prisma=model_cfg.get("use_prisma", False),
        stats_opt_sar_csv=data_cfg["normalization_stats_opt_sar_csv"],
    )

    print(f"Dataset size: {len(dataset)}")

    # --------------------------------------------------
    # Sample one element
    # --------------------------------------------------
    idx = random.randint(0, len(dataset) - 1)
    sample, label, patch_id = dataset[idx]

    print(f"\nSample index: {idx}")
    print(f"Patch ID:     {patch_id}")
    print(f"Label:        {label}")

    # --------------------------------------------------
    # Unpack sample
    # --------------------------------------------------
    if not isinstance(sample, tuple):
        raise RuntimeError("Expected tuple sample (single or fusion mode)")

    tensors = [x for x in sample if x is not None]

    for i, t in enumerate(tensors):
        print(f"Input {i} shape: {tuple(t.shape)}")

    # --------------------------------------------------
    # Visualize inputs (SAR / Planet / PRISMA)
    # --------------------------------------------------
    if model_cfg.get("use_prisma", False):
        prisma = sample[-1]  # always last
        C, H, W = prisma.shape
        print(f"\nPRISMA tensor: {C} channels")

        show_prisma = min(6, C)

        # Determine presence of SAR / Planet
        sar = None
        planet = None

        if dataset.mode == "sar":
            sar = sample[0]
        elif dataset.mode == "planet":
            planet = sample[0]
        elif dataset.mode == "fusion":
            sar, planet = sample[0], sample[1]

        print(
        "SAR mean/std:",
        sar.mean().item(),
        sar.std().item(),
        )

        print(
        "Planet mean/std:",
        planet.mean().item(),
        planet.std().item(),
        )

        n_rows = 1
        if sar is not None:
            n_rows += 1
        if planet is not None:
            n_rows += 1

        fig, axes = plt.subplots(
            n_rows, show_prisma, figsize=(3 * show_prisma, 3 * n_rows)
        )

        if n_rows == 1:
            axes = axes[None, :]

        row = 0

        # ---------------- SAR ----------------
        if sar is not None:
            sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)
            axes[row, 0].imshow(torch.clip(sar[0].cpu(), 0, 1), cmap="gray")
            axes[row, 0].set_title("SAR")
            axes[row, 0].axis("off")
            for j in range(1, show_prisma):
                axes[row, j].axis("off")
            row += 1

        # ---------------- PLANET ----------------
        if planet is not None:
            if planet.shape[0] >= 3:
                rgb = planet[:3].permute(1, 2, 0).cpu()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                axes[row, 0].imshow(torch.clip(rgb, 0, 1))
                axes[row, 0].set_title("Planet RGB")
            else:
                axes[row, 0].imshow(planet[0].cpu(), cmap="gray")
                axes[row, 0].set_title("Planet band 0")
            axes[row, 0].axis("off")
            for j in range(1, show_prisma):
                axes[row, j].axis("off")
            row += 1

        # ---------------- PRISMA ----------------
        for i in range(show_prisma):
            axes[row, i].imshow(prisma[i].cpu(), cmap="viridis")
            axes[row, i].set_title(f"PRISMA {i}")
            axes[row, i].axis("off")

        title = "PRISMA PCA"
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\n✔ Dataset test completed\n")


if __name__ == "__main__":
    main()