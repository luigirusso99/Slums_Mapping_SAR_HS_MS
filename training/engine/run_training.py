# training/engine/run_training.py
# Be sure to setup config files in training/configs/experiments.yaml before running this
import yaml
import torch
from torch.utils.data import DataLoader
import os

from training.dataset.dataset import SlumDataset
from training.models.model_factory import build_model_from_cfg
from training.models.utils import load_backbone_config
from training.engine.trainer import Trainer

def compute_pos_weight(dataset):
    """
    Computes pos_weight = (# negative) / (# positive)
    for BCEWithLogitsLoss from the training dataset labels.
    """
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))

    labels = torch.tensor(labels)
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()

    if pos == 0:
        print("⚠ Warning: No positive samples in training set! pos_weight set to 1.")
        return torch.tensor([1.0], dtype=torch.float32)

    pos_weight = neg / pos
    return torch.tensor([pos_weight], dtype=torch.float32)

def run_fold(fold_id, cfg_path):
    """
    Runs a single training fold using the configuration in cfg_path.
    Supports mode ∈ {sar, planet, fusion} and fusion_type ∈ {early, mid, late}.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"].copy()
    data_cfg = cfg.get("data", {})

    # --------------------------------------------------
    # Inject data-related parameters into model config
    # (required for automatic input channel handling)
    # --------------------------------------------------
    model_cfg["mode"] = data_cfg.get("mode", "fusion")
    model_cfg["sar_channels"] = data_cfg.get("sar_channels")
    model_cfg["planet_channels"] = data_cfg.get("planet_channels")

    # Mode comes from data section
    mode = data_cfg.get("mode", "fusion")
    fusion_type = model_cfg.get("fusion_type", "mid")

    # Ensure consistency: if mode != fusion, fusion_type must be none
    if mode != "fusion":
        model_cfg["fusion_type"] = "none"
        fusion_type = "none"
    elif fusion_type == "none":
        raise ValueError("fusion_type='none' is invalid when mode='fusion'")

    # ------------------------
    # Folds
    # ------------------------
    all_folds = list(range(1, cfg["num_folds"] + 1))
    folds_train = [f for f in all_folds if f != fold_id]
    folds_val = [fold_id]

    # ------------------------
    # Dataset
    # ------------------------
    train_ds = SlumDataset(
        metadata_csv=data_cfg["metadata_csv"],
        folds_csv=data_cfg["folds_csv"],
        folds_to_use=folds_train,
        mode=mode,
        normalize=data_cfg["normalize"],
        augment=data_cfg["augment"] 
    )

    val_ds = SlumDataset(
        metadata_csv=data_cfg["metadata_csv"],
        folds_csv=data_cfg["folds_csv"],
        folds_to_use=folds_val,
        mode=mode,
        normalize=data_cfg["normalize"],
        augment=False
    )

    # ------------------------
    # Dataloaders
    # ------------------------
    # Training dataloader (always standard shuffled dataloader)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
    )

    # Validation → NO balanced sampling
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0)
    )

    # ------------------------
    # Model
    # ------------------------
    backbones_cfg = load_backbone_config()
    from training.models.model_factory import ModelConfig
    model = build_model_from_cfg(backbones_cfg, ModelConfig(**model_cfg))

    # Device selection
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)

    data_mode = cfg["data"]["mode"]  # "sar" | "planet" | "fusion"

    # Compute pos_weight for loss always from training dataset
    pos_weight = compute_pos_weight(train_ds)
    optimizer_cfg = cfg["optimizer"]

    # Consolidated configuration print
    print(f"[Fold {fold_id}] mode={mode} | fusion={fusion_type} | device={device} | pos_weight={pos_weight.item():.3f}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=data_mode,
        optimizer_cfg=optimizer_cfg,
        pos_weight=pos_weight,   # o False se usi sampler
    )

    # ------------------------
    # Training loop
    # ------------------------
    best_f1 = 0.0   # track best F1-score
    epochs = cfg["epochs_per_fold"]

    for epoch in range(epochs):
        (
            tr_loss,
            tr_acc,
            tr_prec,
            tr_rec,
            tr_f1
        ) = trainer.train_one_epoch()

        (
            val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1
        ) = trainer.evaluate()

        print(f"[Fold {fold_id}] Epoch {epoch+1}/{epochs} | Train F1={tr_f1:.3f} | Val F1={val_f1:.3f}", end="\r")

        # Save best model by F1
        checkpoint_dir = f"{cfg['save_dir']}/{model_cfg['mode']}/{model_cfg.get('fusion_type','none')}/{model_cfg['backbone_sar']}/fold_{fold_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(checkpoint_dir, "weights.pt")
            torch.save(model.state_dict(), checkpoint_path)

            # Save metrics corresponding to this best checkpoint
            metrics_path = os.path.join(checkpoint_dir, "final_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Fold {fold_id} BEST Validation Metrics\n")
                f.write(f"Best_F1: {best_f1:.4f}\n")
                f.write(f"Val_Loss: {val_loss:.4f}\n")
                f.write(f"Val_Acc: {val_acc:.4f}\n")
                f.write(f"Val_Prec: {val_prec:.4f}\n")
                f.write(f"Val_Rec: {val_rec:.4f}\n")
                f.write(f"Val_F1: {val_f1:.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="training/configs/experiment.yaml",
                        help="Path to training config file.")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    num_folds = cfg["num_folds"]


    for fold in range(1, num_folds + 1):
        run_fold(fold, cfg_path=args.cfg)

    print("\n✔ Training completed for all folds.")