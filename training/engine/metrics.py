import torch

def compute_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return (preds == labels.long()).float().mean().item()

def compute_precision(logits, labels, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    return (tp / (tp + fp + eps)).item()

def compute_recall(logits, labels, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    return (tp / (tp + fn + eps)).item()

def compute_f1(logits, labels, eps=1e-7):
    precision = compute_precision(logits, labels, eps)
    recall = compute_recall(logits, labels, eps)
    return (2 * precision * recall / (precision + recall + eps))
