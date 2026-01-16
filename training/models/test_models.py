import yaml
import torch

from training.models.model_factory import ModelConfig, build_model_from_cfg
from training.models.utils import list_available_backbones, load_backbone_config


def test_single_configuration(backbones_cfg, backbone_name):
    """
    Builds and tests a MID-FUSION model using the same backbone
    for both PRISMA and PLANET.
    """
    print(f"\n=== Testing MID-FUSION with backbone: {backbone_name} ===")

    cfg = ModelConfig(
        fusion_type="late", # early; mid; late
        backbone_prisma=backbone_name,
        backbone_planet=backbone_name,
        prisma_channels=15,
        planet_channels=8,
        num_classes=2,
        fusion_dim=512,     # can adapt dynamically inside model_factory
        dropout=0.3,
    )

    model = build_model_from_cfg(backbones_cfg, cfg)
    model.eval()

    x_prisma = torch.randn(1, 15, 224, 224)
    x_planet = torch.randn(1, 8, 224, 224)

    with torch.no_grad():
        logits = model(x_prisma, x_planet)

    print(f" → Output logits shape: {logits.shape}")   # [2, 2]


def test_all_backbones():
    """
    Automatically tests all backbones listed inside backbones.yaml
    for MID FUSION (PRISMA + PLANET).
    """
    print("Loading backbone configuration...")
    backbones_cfg = load_backbone_config()
    available = list_available_backbones()

    print("\nAvailable backbones:")
    for name in available:
        print("  •", name)

    for backbone in available:
        try:
            test_single_configuration(backbones_cfg, backbone)
        except Exception as e:
            print(f" !!! ERROR testing backbone {backbone}: {e}")


if __name__ == "__main__":
    # ===============================
    # User‑configurable section
    # ===============================
    USER_FUSION_TYPE = "early"      # options: "early", "mid", "late"
    USER_TEST_MODE = "single"         # "all" or "single"
    USER_BACKBONE = "resnet18"     # used only if USER_TEST_MODE == "single"
    # ===============================

    backbones_cfg = load_backbone_config()

    if USER_TEST_MODE == "all":
        print(f"\n>>> Testing ALL backbones using fusion = {USER_FUSION_TYPE}\n")
        available = list_available_backbones()
        for backbone_name in available:
            try:
                cfg = ModelConfig(
                    fusion_type=USER_FUSION_TYPE,
                    backbone_prisma=backbone_name,
                    backbone_planet=backbone_name,
                    prisma_channels=15,
                    planet_channels=8,
                    num_classes=2,
                    fusion_dim=512,
                    dropout=0.3,
                )
                model = build_model_from_cfg(backbones_cfg, cfg)
                model.eval()
                x_prisma = torch.randn(1, 15, 224, 224)
                x_planet = torch.randn(1, 8, 224, 224)
                with torch.no_grad():
                    logits = model(x_prisma, x_planet)
                print(f"[{backbone_name}] logits shape: {logits.shape}")
            except Exception as e:
                print(f"[{backbone_name}] ERROR: {e}")

    elif USER_TEST_MODE == "single":
        print(f"\n>>> Testing SINGLE backbone = {USER_BACKBONE} using fusion = {USER_FUSION_TYPE}\n")
        cfg = ModelConfig(
            fusion_type=USER_FUSION_TYPE,
            backbone_prisma=USER_BACKBONE,
            backbone_planet=USER_BACKBONE,
            prisma_channels=15,
            planet_channels=8,
            num_classes=2,
            fusion_dim=512,
            dropout=0.3,
        )
        model = build_model_from_cfg(backbones_cfg, cfg)
        model.eval()
        x_prisma = torch.randn(1, 15, 224, 224)
        x_planet = torch.randn(1, 8, 224, 224)
        with torch.no_grad():
            logits = model(x_prisma, x_planet)
        print(f"[{USER_BACKBONE}] logits shape: {logits.shape}")