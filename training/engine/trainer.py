import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from training.engine.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
)

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer_cfg=None, pos_weight=None):
        # Auto-select device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.pos_weight = pos_weight

        # Loss function selection: BCE with optional pos_weight
        if self.pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        else:
            print("→ Using BCEWithLogitsLoss (NO pos_weight)")
            self.criterion = nn.BCEWithLogitsLoss()

        if optimizer_cfg is None:
            raise ValueError("optimizer_cfg must be provided")

        opt_type = optimizer_cfg["type"].lower()
        lr = float(optimizer_cfg["lr"])
        wd = float(optimizer_cfg["weight_decay"])

        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    # ============================
    # Internal helper
    # ============================
    def forward_pass(self, inputs, labels):
        """
        Fully data-driven forward pass.
        The dataset defines the structure of `inputs`.
        The model signature must match it.
        """
        labels = labels.to(self.device).float().unsqueeze(1)

        # --------------------
        # DATA-DRIVEN FORWARD
        # --------------------
        if isinstance(inputs, (tuple, list)):
            inputs = [x.to(self.device) for x in inputs]
            logits = self.model(*inputs)
        else:
            logits = self.model(inputs.to(self.device))

        # --------------------
        # LOSS + METRICS
        # --------------------
        loss = self.criterion(logits, labels)

        logits_flat = logits.squeeze(1)
        labels_flat = labels.squeeze(1)

        acc = compute_accuracy(logits_flat, labels_flat)
        prec = compute_precision(logits_flat, labels_flat)
        rec = compute_recall(logits_flat, labels_flat)
        f1 = compute_f1(logits_flat, labels_flat)

        return loss, acc, prec, rec, f1

    # ============================
    # TRAINING
    # ============================
    def train_one_epoch(self):
        self.model.train()
        
        total_loss = total_acc = total_prec = total_rec = total_f1 = 0.0

        train_iter = tqdm(self.train_loader, desc="Training", leave=False)

        for inputs, labels in train_iter:
            self.optimizer.zero_grad()

            loss, acc, prec, rec, f1 = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            total_prec += prec
            total_rec += rec
            total_f1 += f1

            train_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.3f}",
                "prec": f"{prec:.3f}",
                "rec": f"{rec:.3f}",
                "f1": f"{f1:.3f}",
            })

        n = len(self.train_loader)
        return (
            total_loss / n,
            total_acc / n,
            total_prec / n,
            total_rec / n,
            total_f1 / n,
        )

    # ============================
    # VALIDATION
    # ============================
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        total_loss = total_acc = total_prec = total_rec = total_f1 = 0.0

        val_iter = tqdm(self.val_loader, desc="Validation", leave=False)

        for inputs, labels in val_iter:
            loss, acc, prec, rec, f1 = self.forward_pass(inputs, labels)

            total_loss += loss.item()
            total_acc += acc
            total_prec += prec
            total_rec += rec
            total_f1 += f1

            val_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.3f}",
                "prec": f"{prec:.3f}",
                "rec": f"{rec:.3f}",
                "f1": f"{f1:.3f}",
            })

        n = len(self.val_loader)
        return (
            total_loss / n,
            total_acc / n,
            total_prec / n,
            total_rec / n,
            total_f1 / n,
        )