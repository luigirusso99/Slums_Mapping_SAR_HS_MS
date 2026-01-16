from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict

import torch.nn as nn

from .feature_extractors import build_encoder_from_cfg  
from .fusion_strategies import (
    SingleSensorModel,
    EarlyFusionModel,
    MidFusionModel,
    LateFusionModel,
)

FusionType = Literal["single", "early", "mid", "late"]

@dataclass
class ModelConfig:
    # ---- REQUIRED (non-default) ----
    fusion_type: FusionType           # "single", "early", "mid", "late"
    mode: str                         # "sar", "planet", "fusion"
    backbone_sar: str              # name from backbones.yaml

    # ---- OPTIONAL (with defaults) ----
    backbone_planet: Optional[str] = None  # if None, reuse backbone_sar
    num_classes: int = 2
    sar_channels: Optional[int] = None
    planet_channels: Optional[int] = None
    fusion_dim: int = 512
    dropout: float = 0.2
    pool_type: str = "avg"
    alpha_late: float = 0.5
    pretrained: bool = True
    library_sar: str = "torchvision"
    library_planet: str = "torchvision"


def build_model_from_cfg(backbones_cfg: Dict, cfg: ModelConfig) -> nn.Module:
    """
    backbones_cfg: dict loaded from backbones.yaml
    cfg: ModelConfig con tutte le info.

    Ritorna un nn.Module che restituisce logits [B, num_classes].
    """
    fusion = cfg.fusion_type
    mode = cfg.mode

    # Resolve channel counts (must be provided from experiment.yaml)
    if cfg.mode in ("sar", "fusion"):
        if cfg.sar_channels is None:
            raise ValueError("sar_channels must be set in experiment.yaml for mode='sar' or 'fusion'")
        sar_channels = cfg.sar_channels
    else:
        sar_channels = None

    if cfg.mode in ("planet", "fusion"):
        if cfg.planet_channels is None:
            raise ValueError("planet_channels must be set in experiment.yaml for mode='planet' or 'fusion'")
        planet_channels = cfg.planet_channels
    else:
        planet_channels = None

    # Helper: trova entry nel backbones.yaml
    def _find_backbone_entry(name: str) -> dict:
        for entry in backbones_cfg["backbones"]:
            if entry["name"] == name:
                return entry
        raise ValueError(f"Backbone '{name}' not found in backbones.yaml")

    # -----------------------------------------------
    # MODE-FIRST LOGIC  (sar | planet | fusion)
    # -----------------------------------------------

    # ========== SINGLE-SENSOR: sar ==========
    if mode == "sar":
        entry = _find_backbone_entry(cfg.backbone_sar)

        encoder = build_encoder_from_cfg(
            entry,
            in_channels=sar_channels,
        )

        model = SingleSensorModel(
            encoder=encoder,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )
        model.model_mode = "sar"
        return model

    # ========== SINGLE-SENSOR: PLANET ==========
    if mode == "planet":
        # allow custom backbone for PLANET, fallback to sar backbone
        bb_planet = cfg.backbone_planet or cfg.backbone_sar
        entry = _find_backbone_entry(bb_planet)

        encoder = build_encoder_from_cfg(
            entry,
            in_channels=planet_channels,
        )

        model = SingleSensorModel(
            encoder=encoder,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )
        model.model_mode = "planet"
        return model

    # ========== FUSION (requires fusion_type) ==========
    if mode != "fusion":
        raise ValueError(
            f"Invalid mode '{mode}'. Expected: sar | planet | fusion."
        )

    elif fusion == "early":
        # Un solo encoder che riceve [B, C_p + C_q, H, W]
        total_channels = sar_channels + planet_channels
        entry = _find_backbone_entry(cfg.backbone_sar)
        enc = build_encoder_from_cfg(
            entry,
            in_channels=total_channels,
        )
        model = EarlyFusionModel(
            encoder=enc,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )
        return model

    elif fusion == "mid":
        # Due encoder, uno per sar, uno per Planet
        entry_p = _find_backbone_entry(cfg.backbone_sar)
        b_planet = cfg.backbone_planet or cfg.backbone_sar
        entry_q = _find_backbone_entry(b_planet)

        enc_p = build_encoder_from_cfg(
            entry_p,
            in_channels=sar_channels,
        )
        enc_q = build_encoder_from_cfg(
            entry_q,
            in_channels=planet_channels,
        )

        model = MidFusionModel(
            encoder_sar=enc_p,
            encoder_planet=enc_q,
            fusion_dim=cfg.fusion_dim,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )
        return model

    elif fusion == "late":
        # Due SingleSensorModel, fusi a livello di logits
        entry_p = _find_backbone_entry(cfg.backbone_sar)
        b_planet = cfg.backbone_planet or cfg.backbone_sar
        entry_q = _find_backbone_entry(b_planet)

        enc_p = build_encoder_from_cfg(
            entry_p,
            in_channels=sar_channels,
        )
        enc_q = build_encoder_from_cfg(
            entry_q,
            in_channels=planet_channels,
        )

        model_p = SingleSensorModel(
            encoder=enc_p,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )
        model_q = SingleSensorModel(
            encoder=enc_q,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            pool_type=cfg.pool_type,
        )

        model = LateFusionModel(
            model_sar=model_p,
            model_planet=model_q,
            alpha=cfg.alpha_late,
        )
        return model

    else:
        raise ValueError(f"Unknown fusion_type: {fusion} (mode={mode})")