from training.models.fusion_strategies import (
    SingleSensorModel,
    EarlyFusion,
    MidFusion,
    LateFusion,
)


def build_model(cfg: dict):
    """
    Factory function to build a model according to experiment.yaml.

    This factory is aligned with the new PRISMA PCA embedding strategy:
        - PRISMA is NEVER used for conditioning or FiLM
        - PRISMA is embedded separately and concatenated at feature level
        - ResNet18 backbones are always pure
    """

    fusion = cfg["fusion_type"]
    dropout = cfg.get("dropout", 0.2)
    pool = cfg.get("pool_type", "avg")
    pretrained = cfg.get("pretrained", True)

    use_prisma = cfg.get("use_prisma", False)
    prisma_channels = cfg.get("prisma_channels") if use_prisma else None
    prisma_emb_dim = cfg.get("prisma_emb_dim", 128)

    num_classes = cfg["num_classes"]

    # ==========================
    # SINGLE-SENSOR BASELINES
    # ==========================
    if fusion == "single":
        sensor = cfg["sensor_type"]  # "sar" | "planet"

        if sensor == "sar":
            in_ch = cfg["sar_channels"]
        elif sensor == "planet":
            in_ch = cfg["planet_channels"]
        else:
            raise ValueError(f"Unknown sensor_type: {sensor}")

        return SingleSensorModel(
            in_channels=in_ch,
            num_classes=num_classes,
            prisma_channels=prisma_channels,
            prisma_emb_dim=prisma_emb_dim,
            dropout=dropout,
            pool_type=pool,
            pretrained=pretrained,
        )

    # ==========================
    # EARLY FUSION
    # ==========================
    if fusion == "early":
        in_ch = cfg["sar_channels"] + cfg["planet_channels"]

        return EarlyFusion(
            in_channels=in_ch,
            num_classes=num_classes,
            prisma_channels=prisma_channels,
            prisma_emb_dim=prisma_emb_dim,
            dropout=dropout,
            pool_type=pool,
            pretrained=pretrained,
        )

    # ==========================
    # MID FUSION
    # ==========================
    if fusion == "mid":
        return MidFusion(
            sar_channels=cfg["sar_channels"],
            planet_channels=cfg["planet_channels"],
            num_classes=num_classes,
            prisma_channels=prisma_channels,
            prisma_emb_dim=prisma_emb_dim,
            dropout=dropout,
            pool_type=pool,
            pretrained=pretrained,
        )

    # ==========================
    # LATE FUSION
    # ==========================
    if fusion == "late":
        return LateFusion(
            sar_channels=cfg["sar_channels"],
            planet_channels=cfg["planet_channels"],
            num_classes=num_classes,
            prisma_channels=prisma_channels,
            prisma_emb_dim=prisma_emb_dim,
            dropout=dropout,
            pool_type=pool,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown fusion_type: {fusion}")