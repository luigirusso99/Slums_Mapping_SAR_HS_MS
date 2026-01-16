# training/models/utils.py

from __future__ import annotations
import yaml
from typing import List


def load_backbone_config(path: str = "training/configs/backbones.yaml") -> dict:
    """
    Load the YAML file containing backbone configurations.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def list_available_backbones(path: str = "training/configs/backbones.yaml") -> List[str]:
    """
    Returns the list of valid backbone names defined inside backbones.yaml.
    """
    cfg = load_backbone_config(path)
    return [entry["name"] for entry in cfg["backbones"]]