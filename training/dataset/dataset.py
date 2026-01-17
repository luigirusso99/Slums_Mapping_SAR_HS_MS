# training/dataset.py

import torch, rasterio, pandas as pd, random, numpy as np
from torch.utils.data import Dataset
from training.dataset.utils.augmentations import GeoAugmentations
from training.dataset.utils.normalization import ZScoreNormalizer

class SlumDataset(Dataset):

    def __init__(self, metadata_csv, folds_csv, folds_to_use,
                 fusion_type="fusion",
                 sensor_type=None,
                 normalize=True, augment=True, return_id=False,
                 use_prisma=False,
                 stats_opt_sar_csv="dataset/normalization_stats.csv",
                 stats_prisma_csv=None):
        
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

        # internal augmentations
        self.aug = GeoAugmentations() if augment else None

        # normalizers are now derived automatically from min/max statistics
        if normalize:
            self.norm_sar, self.norm_planet, self.norm_prisma = self.load_stats_and_build_normalizers_multi(
                stats_opt_sar_csv, stats_prisma_csv
            )
        else:
            self.norm_sar = None
            self.norm_planet = None
            self.norm_prisma = None

    def load_stats_and_build_normalizers_multi(self, stats_opt_sar_csv, stats_prisma_csv):
        """
        Loads per-band statistics from normalization CSVs and builds
        Z-score normalizers for:
        - SAR (single-band backscatter)
        - PlanetScope (8-band multispectral)
        - PRISMA (if use_prisma=True)
        Note: SAR and Planet statistics are read from the same CSV using the 'sensor' column.
        PRISMA statistics are read from a separate CSV.

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

        # Load PRISMA stats only if use_prisma=True
        norm_prisma = None
        if self.use_prisma:
            if stats_prisma_csv is None:
                raise RuntimeError("stats_prisma_csv must be provided if use_prisma=True and normalize=True")

            prisma_df = pd.read_csv(stats_prisma_csv)
            prisma_df = prisma_df[prisma_df["sensor"] == "prisma"].sort_values("band")

            mean_prisma = torch.tensor(prisma_df["mean"].values, dtype=torch.float32)
            std_prisma  = torch.tensor(prisma_df["std"].values,  dtype=torch.float32)
            std_prisma[std_prisma == 0] = 1.0
            norm_prisma = ZScoreNormalizer(mean_prisma, std_prisma)

        return norm_sar, norm_planet, norm_prisma
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

        # Augmentations only on sar and planet
        if self.aug:
            sar, planet = self.aug(sar, planet)

        # Normalization
        if self.normalize:
            if sar is not None and self.norm_sar is not None:
                sar = self.norm_sar(sar)
            if planet is not None and self.norm_planet is not None:
                planet = self.norm_planet(planet)
            if prisma is not None and self.norm_prisma is not None:
                prisma = self.norm_prisma(prisma)

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
    
if __name__ == "__main__":
    # Flexible test for all modes
    dataset = SlumDataset(
        metadata_csv="dataset/metadata.csv",
        folds_csv="dataset/folds.csv",
        folds_to_use=[1,2,3],
        fusion_type="single",
        sensor_type="sar",
        normalize=True,
        augment=True,
        return_id=True,
        use_prisma=True,
        stats_opt_sar_csv="dataset/normalization_stats_opt_sar.csv",
        stats_prisma_csv="dataset/prisma_normalization_stats.csv"
    )

    print("Dataset size:", len(dataset))
    idx = random.randint(0, len(dataset)-1)

    out = dataset[idx]

    if len(out) == 3:
        sample, label, pid = out
        print("Patch ID:", pid)
    else:
        sample, label = out

    # Detect mode (single vs fusion)
    if isinstance(sample, tuple):
        if dataset.use_prisma and len(sample) == 3:
            sar, planet, prisma = sample
            print("sar shape:", sar.shape if sar is not None else None)
            print("Planet shape:", planet.shape if planet is not None else None)
            print("Prisma shape:", prisma.shape if prisma is not None else None)
            print(torch.unique(sar), torch.unique(planet), torch.unique(prisma))
            import matplotlib.pyplot as plt
            plt.subplot(1, 3, 1)
            plt.imshow(sar[0], cmap = 'gray')
            plt.subplot(1, 3, 2)
            plt.imshow(planet[0], cmap = 'gray')
            plt.subplot(1, 3, 3)
            plt.imshow(prisma[0], cmap = 'gray')
            plt.show()
        elif dataset.use_prisma and len(sample) == 2:
            # this covers cases sar+prisma or planet+prisma
            a, b = sample
            print("Sample 1 shape:", a.shape if a is not None else None)
            print("Sample 2 shape:", b.shape if b is not None else None)
            print(torch.unique(a), torch.unique(b))
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(a[0], cmap = 'gray')
            plt.subplot(1, 2, 2)
            plt.imshow(b[0], cmap = 'gray')
            plt.show()
        else:
            sar, planet = sample
            print("sar shape:", sar.shape if sar is not None else None)
            print("Planet shape:", planet.shape if planet is not None else None)
            print(torch.unique(sar), torch.unique(planet))
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(sar[0], cmap = 'gray')
            plt.subplot(1, 2, 2)
            plt.imshow(planet[0], cmap = 'gray')
            plt.show()
        
    else:
        print("Sample shape:", sample.shape)

    print("Label:", label)

    # Additional example: fusion dataset
    fusion_dataset = SlumDataset(
        metadata_csv="dataset/metadata.csv",
        folds_csv="dataset/folds.csv",
        folds_to_use=[1,2,3],
        fusion_type="mid",
        sensor_type=None,
        normalize=True,
        augment=True,
        return_id=True,
        use_prisma=True,
        stats_opt_sar_csv="dataset/normalization_stats_opt_sar.csv",
        stats_prisma_csv="dataset/prisma_normalization_stats.csv"
    )

    print("Fusion dataset size:", len(fusion_dataset))
    idx = random.randint(0, len(fusion_dataset)-1)

    out = fusion_dataset[idx]

    if len(out) == 3:
        sample, label, pid = out
        print("Patch ID:", pid)
    else:
        sample, label = out

    if isinstance(sample, tuple):
        if fusion_dataset.use_prisma and len(sample) == 3:
            sar, planet, prisma = sample
            print("sar shape:", sar.shape if sar is not None else None)
            print("Planet shape:", planet.shape if planet is not None else None)
            print("Prisma shape:", prisma.shape if prisma is not None else None)
        elif fusion_dataset.use_prisma and len(sample) == 2:
            a, b = sample
            print("Sample 1 shape:", a.shape if a is not None else None)
            print("Sample 2 shape:", b.shape if b is not None else None)
        else:
            sar, planet = sample
            print("sar shape:", sar.shape if sar is not None else None)
            print("Planet shape:", planet.shape if planet is not None else None)
    else:
        print("Sample shape:", sample.shape)

    print("Label:", label)