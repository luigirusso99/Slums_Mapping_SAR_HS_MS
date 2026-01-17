import torch
from training.models.model_factory import build_model

def run(cfg):
    model = build_model(cfg)
    model.eval()

    x_s = torch.randn(1, cfg["sar_channels"], 224, 224)
    x_p = torch.randn(1, cfg["planet_channels"], 224, 224)
    x_h = torch.randn(1, cfg["prisma_channels"], 224, 224) if "prisma_channels" in cfg else None

    with torch.no_grad():
        if cfg["fusion_type"] == "single":
            if cfg["sensor_type"] == "sar":
                y = model(x_s)
            else:
                y = model(x_p)

        elif cfg["fusion_type"] == "late":
            y = model(x_s, x_p, x_h)

        else:
            y = model(torch.cat([x_s, x_p], dim=1), x_h)

    print(cfg, "→", y.shape)


if __name__ == "__main__":

    base = dict(
        sar_channels=2,
        planet_channels=8,
        num_classes=2,
        dropout=0.3,
        pool_type="avg",
    )

    # -------- SINGLE SENSOR --------
    run({**base, "fusion_type": "single", "sensor_type": "sar"})
    run({**base, "fusion_type": "single", "sensor_type": "planet"})

    # -------- FUSION (no PRISMA) --------
    for f in ["early", "mid", "late"]:
        run({**base, "fusion_type": f})

    # -------- FUSION + PRISMA --------
    for f in ["early", "mid", "late"]:
        run({**base, "fusion_type": f, "prisma_channels": 15})