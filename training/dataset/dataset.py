# training/dataset.py

import torch, rasterio, pandas as pd, random
from torch.utils.data import Dataset
from training.dataset.utils.augmentations import GeoAugmentations
from training.dataset.utils.normalization import ZScoreNormalizer

class SlumDataset(Dataset):

    def __init__(self, metadata_csv, folds_csv, folds_to_use,
                 mode="fusion", normalize=True, augment=True, return_id=False):
        
        assert mode in ("sar", "planet", "fusion")

        meta = pd.read_csv(metadata_csv)
        folds = pd.read_csv(folds_csv)

        df = meta.merge(folds[["patch_id", "fold"]], on="patch_id", how="inner")
        df = df[df["fold"].isin(folds_to_use)]

        if df.empty:
            raise RuntimeError("No patches match selected folds!")

        self.df = df.reset_index(drop=True)

        self.mode = mode
        self.normalize = normalize
        self.augment = augment
        self.return_id = return_id

        # internal augmentations
        self.aug = GeoAugmentations() if augment else None

        # normalizers are now derived automatically from min/max statistics
        if normalize:
            self.norm_sar, self.norm_planet = self.load_stats_and_build_normalizers(
                "dataset/normalization_stats.csv"
            )

    def load_stats_and_build_normalizers(self, csv_path):
        """
        Loads per-band statistics from normalization_stats.csv and builds
        Z-score normalizers for:
        - SAR (single-band backscatter)
        - PlanetScope (8-band multispectral)
        """

        df = pd.read_csv(csv_path)

        # --- Extract sar stats ---
        sar_df = df[df["sensor"] == "sar"].sort_values("band")
        assert len(sar_df) == 1, f"Expected 1 SAR band, got {len(sar_df)}"
        planet_df = df[df["sensor"] == "planet"].sort_values("band")

        mean_sar = torch.tensor(sar_df["mean"].values, dtype=torch.float32)
        std_sar  = torch.tensor(sar_df["std"].values,  dtype=torch.float32)

        mean_planet = torch.tensor(planet_df["mean"].values, dtype=torch.float32)
        std_planet  = torch.tensor(planet_df["std"].values,  dtype=torch.float32)

        # safety: avoid division by zero
        std_sar[std_sar == 0] = 1.0
        std_planet[std_planet == 0] = 1.0

        norm_sar = ZScoreNormalizer(mean_sar, std_sar)
        norm_planet = ZScoreNormalizer(mean_planet, std_planet)

        return norm_sar, norm_planet
    # ------------------------------

    def load(self, path):
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")
        return torch.tensor(arr)

    # ------------------------------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sar = self.load(row["sar_path"]) if self.mode in ("sar", "fusion") else None
        planet = self.load(row["planet_path"]) if self.mode in ("planet", "fusion") else None

        # Augmentations
        if self.aug:
            sar, planet = self.aug(sar, planet)

        # Normalization
        if self.normalize:
            if sar is not None: sar = self.norm_sar(sar)
            if planet is not None: planet = self.norm_planet(planet)

        y = torch.tensor(row["label"], dtype=torch.long)
        patch_id = row["patch_id"]

        if self.mode == "sar":
            if self.return_id:
                return sar, y, patch_id
            else:
                return sar, y
        if self.mode == "planet":
            if self.return_id:
                return planet, y, patch_id
            else:
                return planet, y
        if self.return_id:
            return (sar, planet), y, patch_id
        else:
            return (sar, planet), y

    def __len__(self):
        return len(self.df)
    
if __name__ == "__main__":
    # Flexible test for all modes
    dataset = SlumDataset(
        metadata_csv="dataset/metadata.csv",
        folds_csv="dataset/folds.csv",
        folds_to_use=[1,2,3],
        mode="fusion",   # sar | planet | fusion
        normalize=True,
        augment=True,
        return_id=True   #
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