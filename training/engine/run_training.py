# training/engine/run_training.py
# Be sure to setup config files in training/configs/experiments.yaml before running this
import yaml
import torch
from torch.utils.data import DataLoader
import os
from training.dataset.dataset import SlumDataset
from training.models.model_factory import build_model
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
    # =====================
    # Load config
    # =====================
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # =====================
    # Dataset
    # =====================
    # ---------------------
    # Cross-validation folds
    # ---------------------
    all_folds = list(range(1, cfg["num_folds"] + 1))
    folds_train = [f for f in all_folds if f != fold_id]
    folds_val = [fold_id]

    train_ds = SlumDataset(
        metadata_csv=data_cfg["metadata_csv"],
        folds_csv=data_cfg["folds_csv"],
        folds_to_use=folds_train,
        fusion_type=model_cfg["fusion_type"],
        sensor_type=model_cfg.get("sensor_type"),
        normalize=data_cfg.get("normalize", True),
        augment=data_cfg.get("augment", True),
        use_prisma=model_cfg.get("use_prisma", False),
        stats_opt_sar_csv=data_cfg["normalization_stats_opt_sar_csv"],
        stats_prisma_csv=data_cfg.get("normalization_stats_prisma_csv"),
    )

    val_ds = SlumDataset(
        metadata_csv=data_cfg["metadata_csv"],
        folds_csv=data_cfg["folds_csv"],
        folds_to_use=folds_val,
        fusion_type=model_cfg["fusion_type"],
        sensor_type=model_cfg.get("sensor_type"),
        normalize=data_cfg.get("normalize", True),
        augment=False,
        use_prisma=model_cfg.get("use_prisma", False),
        stats_opt_sar_csv=data_cfg["normalization_stats_opt_sar_csv"],
        stats_prisma_csv=data_cfg.get("normalization_stats_prisma_csv"),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # =====================
    # Model
    # =====================
    model = build_model({
        **model_cfg,
        "sar_channels": data_cfg["sar_channels"],
        "planet_channels": data_cfg["planet_channels"],
        "prisma_channels": data_cfg.get("prisma_channels"),
    })

    # =====================
    # Trainer
    # =====================
    pos_weight = compute_pos_weight(train_ds)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_cfg=cfg["optimizer"],
        pos_weight=pos_weight,
    )

    fusion = model_cfg["fusion_type"]
    sensor = model_cfg.get("sensor_type", "fusion")
    prisma_flag = "prisma" if model_cfg.get("use_prisma", False) else "no_prisma"

    device = trainer.device

    print(
        f"[Fold {fold_id}] "
        f"fusion={fusion} | sensor={sensor} | {prisma_flag} | "
        f"device={device} | pos_weight={pos_weight.item():.3f}"
    )

    best_f1 = 0.0
    epochs = cfg["epochs_per_fold"]

    # --------------------
    # Checkpoint directory
    # --------------------
    path_parts = [
        cfg["save_dir"],
        fusion,
    ]

    if fusion == "single":
        path_parts.append(sensor)

    path_parts.extend([
        prisma_flag,
        f"fold_{fold_id}",
    ])

    checkpoint_dir = os.path.join(*path_parts)
    os.makedirs(checkpoint_dir, exist_ok=True)

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

        print(
            f"[Fold {fold_id}] Epoch {epoch+1}/{epochs} | "
            f"Train F1={tr_f1:.3f} | Val F1={val_f1:.3f}"
        )

        # --------------------
        # Save best checkpoint
        # --------------------
        if val_f1 > best_f1:
            best_f1 = val_f1

            checkpoint_path = os.path.join(checkpoint_dir, "weights.pt")
            torch.save(model.state_dict(), checkpoint_path)

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
    parser.add_argument("--cfg", type=str, default='training/configs/experiment.yaml')
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    for fold in range(1, cfg["num_folds"] + 1):
        run_fold(fold, cfg_path=args.cfg)