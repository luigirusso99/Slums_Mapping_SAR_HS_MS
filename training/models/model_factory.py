#training/models/model_factory.py

from training.models.fusion_strategies import (
    SingleSensorModel,
    EarlyFusion,
    MidFusion,
    LateFusion,
)

def build_model(cfg: dict):

    fusion = cfg["fusion_type"]
    dropout = cfg.get("dropout", 0.2)
    pool = cfg.get("pool_type", "avg")
    pretrained = cfg.get("pretrained", True)

    # ==========================
    # SINGLE-SENSOR BASELINES
    # ==========================
    if fusion == "single":
        sensor = cfg["sensor_type"]   # "sar" | "planet"

        if sensor == "sar":
            in_ch = cfg["sar_channels"]
        elif sensor == "planet":
            in_ch = cfg["planet_channels"]
        else:
            raise ValueError(f"Unknown sensor_type: {sensor}")

        use_prisma = cfg.get("use_prisma", False)
        prisma_channels = cfg.get("prisma_channels") if use_prisma else None

        return SingleSensorModel(
            in_channels=in_ch,
            num_classes=cfg["num_classes"],
            prisma_channels=prisma_channels,
            dropout=dropout,
            pool_type=pool,
            pretrained=pretrained,
        )

    # ==========================
    # EARLY FUSION
    # ==========================
    if fusion == "early":
        return EarlyFusion(
            cfg["sar_channels"] + cfg["planet_channels"],
            cfg["num_classes"],
            cfg.get("prisma_channels"),
            dropout,
            pool,
            pretrained,
        )

    # ==========================
    # MID FUSION
    # ==========================
    if fusion == "mid":
        return MidFusion(
            cfg["sar_channels"] + cfg["planet_channels"],
            cfg["num_classes"],
            cfg.get("prisma_channels"),
            dropout,
            pool,
            pretrained,
        )

    # ==========================
    # LATE FUSION
    # ==========================
    if fusion == "late":
        return LateFusion(
            cfg["sar_channels"],
            cfg["planet_channels"],
            cfg["num_classes"],
            cfg.get("prisma_channels"),
            dropout,
            pool,
            pretrained,
        )

    raise ValueError(f"Unknown fusion_type: {fusion}")