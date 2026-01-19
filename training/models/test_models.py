import torch
from training.models.model_factory import build_model


def run(cfg):
    model = build_model(cfg)
    model.eval()

    B = 1
    H = W = 224

    x_s = torch.randn(B, cfg["sar_channels"], H, W)
    x_p = torch.randn(B, cfg["planet_channels"], H, W)

    # PRISMA PCA patches are small (e.g. 23x23)
    x_h = None
    if cfg.get("use_prisma", False):
        C = cfg["prisma_channels"]
        Hp = Wp = cfg.get("prisma_patch_size", 23)
        x_h = torch.randn(B, C, Hp, Wp)

    with torch.no_grad():
        if cfg["fusion_type"] == "single":
            if cfg["sensor_type"] == "sar":
                y = model(x_s, x_h)
            else:
                y = model(x_p, x_h)

        elif cfg["fusion_type"] == "early":
            y = model(x_s, x_p, x_h)

        elif cfg["fusion_type"] == "mid":
            y = model(x_s, x_p, x_h)

        elif cfg["fusion_type"] == "late":
            y = model(x_s, x_p, x_h)

        else:
            raise ValueError(f"Unknown fusion_type: {cfg['fusion_type']}")

    print(cfg, "→", y.shape)


if __name__ == "__main__":

    base = dict(
        sar_channels=2,
        planet_channels=8,
        num_classes=2,
        dropout=0.3,
        pool_type="avg",
        prisma_emb_dim=128,
        pretrained=False,
    )

    # -------- SINGLE SENSOR (no PRISMA) --------
    run({
        **base,
        "fusion_type": "single",
        "sensor_type": "sar",
        "use_prisma": False,
    })

    run({
        **base,
        "fusion_type": "single",
        "sensor_type": "planet",
        "use_prisma": False,
    })

    # -------- FUSION (no PRISMA) --------
    for f in ["early", "mid", "late"]:
        run({
            **base,
            "fusion_type": f,
            "use_prisma": False,
        })

    # -------- FUSION + PRISMA PCA --------
    for f in ["early", "mid", "late"]:
        run({
            **base,
            "fusion_type": f,
            "use_prisma": True,
            "prisma_channels": 16,
            "prisma_patch_size": 23,
        })