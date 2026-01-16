# training/models/feature_extractors.py

import yaml
import timm
import torch
import torch.nn as nn
from torchvision import models as tvm

class TimmWrapper(nn.Module):
    def __init__(self, model, return_multiscale=False):
        super().__init__()
        self.model = model
        self.return_multiscale = return_multiscale

    def forward(self, x):
        out = self.model(x)

        # Multi-scale mode → return full list with normalization
        if self.return_multiscale:
            processed = []
            feats = out if isinstance(out, list) else [out]
            for feat in feats:
                if isinstance(feat, torch.Tensor) and feat.ndim == 4 and feat.shape[-1] != feat.shape[1]:
                    # NHWC → NCHW
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                processed.append(feat)
            return processed

        # Single-scale mode → use last feature if list
        if isinstance(out, list):
            out = out[-1]

        # Convert NHWC → NCHW if needed
        if isinstance(out, torch.Tensor) and out.ndim == 4 and out.shape[-1] != out.shape[1]:
            out = out.permute(0, 3, 1, 2).contiguous()

        return out

def patch_input_conv(model: nn.Module, in_channels: int) -> nn.Module:
    """
    Replace the input conv layer of a torchvision / timm model to accept arbitrary in_channels.
    Keeps original out_channels, kernel_size, stride, padding, bias.
    Note: weights for new channels are re-initialized.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            old = module
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None)
            )
            parent = model
            path = name.split('.')
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], new)
            return model
    raise RuntimeError("No Conv2d layer found to patch for input_conv")

def load_backbones_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("backbones", [])

def find_first_conv(module: nn.Module):
    """
    Recursively finds the first nn.Conv2d in any torchvision model.
    """
    for child in module.children():
        if isinstance(child, nn.Conv2d):
            return child
        result = find_first_conv(child)
        if result is not None:
            return result
    return None


def patch_first_conv(module: nn.Module, in_channels: int):
    """
    Replaces the first Conv2d in the model with a new Conv2d that accepts
    in_channels while copying pretrained weights smartly.
    """
    target_conv = find_first_conv(module)
    if target_conv is None:
        raise RuntimeError("Could not locate first Conv2d layer in model.")

    old_weight = target_conv.weight.data
    old_in_ch = old_weight.shape[1]
    out_ch = old_weight.shape[0]

    # Create new conv with same hyperparameters but new input channels
    new_conv = nn.Conv2d(
        in_channels,
        out_ch,
        kernel_size=target_conv.kernel_size,
        stride=target_conv.stride,
        padding=target_conv.padding,
        dilation=target_conv.dilation,
        bias=target_conv.bias is not None,
    )

    # Smart weight initialization
    if in_channels == old_in_ch:
        new_conv.weight.data = old_weight
    elif in_channels < old_in_ch:
        # Use first channels
        new_conv.weight.data = old_weight[:, :in_channels]
    else:
        # Replicate to fill channels
        repeat = (in_channels // old_in_ch) + 1
        new_conv.weight.data = old_weight.repeat(1, repeat, 1, 1)[:, :in_channels]

    if target_conv.bias is not None:
        new_conv.bias.data = target_conv.bias.data

    # Replace inside parent module
    def replace_conv(parent, target, new):
        for name, child in parent.named_children():
            if child is target:
                setattr(parent, name, new)
                return True
            if replace_conv(child, target, new):
                return True
        return False

    replace_conv(module, target_conv, new_conv)

class UnifiedBackbone(nn.Module):
    """
    Forces all backbones to provide consistent output:
        - Always returns [B, C, H, W]
        - Applies adaptive pooling if features are flattened
        - Fixes NHWC vs NCHW
    """

    def __init__(self, model, out_channels: int):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        out = self.model(x)

        # A) TIMM Transformers may output channel-last [B, H, W, C]
        if isinstance(out, torch.Tensor) and out.ndim == 4 and out.shape[-1] == self.out_channels:
            out = out.permute(0, 3, 1, 2)

        # B) Flattened features [B, C]
        if isinstance(out, torch.Tensor) and out.ndim == 2:
            out = out.unsqueeze(-1).unsqueeze(-1)  # → [B, C, 1, 1]

        # C) Ensure 4D output
        if out.ndim != 4:
            raise RuntimeError(f"Backbone output has unsupported shape: {out.shape}")

        return out

def build_encoder_from_cfg(cfg_entry, in_channels=3):
    """
    Costruisce un encoder a partire dal dizionario YAML.
    Si occupa di:
      - costruire il modello torchvision o timm
      - adattare il primo layer a in_channels diversi
      - sistemare i casi speciali (Swin channel-last)
      - assegnare encoder.out_channels = valore YAML
    """

    name = cfg_entry["name"]
    lib = cfg_entry["library"]
    pretrained = cfg_entry.get("pretrained", True)
    out_ch = cfg_entry.get("out_channels", None)

    if out_ch is None:
        raise ValueError(f"[feature_extractors] Missing out_channels for backbone: {name}")

    # ===========================
    #  TORCHVISION MODELS
    # ===========================
    if lib == "torchvision":
        if not hasattr(tvm, name):
            raise ValueError(f"[feature_extractors] Torchvision has no model '{name}'")

        model_fn = getattr(tvm, name)

        # load pretrained
        try:
            model = model_fn(weights="DEFAULT" if pretrained else None)
        except:
            model = model_fn(pretrained=pretrained)

        # Patch ANY CNN backbone first conv to accept N channels
        if in_channels != 3:
            patch_first_conv(model, in_channels)

        encoder = model

        # Remove classification layers if exist
        if hasattr(encoder, "fc"):
            encoder.fc = nn.Identity()
        if hasattr(encoder, "classifier"):
            encoder.classifier = nn.Identity()

            # Remove classifier head
            if hasattr(encoder, "fc"):
                encoder.fc = nn.Identity()
            if hasattr(encoder, "classifier"):
                encoder.classifier = nn.Identity()

    # ===========================
    # TIMM MODELS  (e.g., SWIN)
    # ===========================
    elif lib == "timm":
        import timm

        if name not in timm.list_models():
            raise ValueError(f"[feature_extractors] Model '{name}' not found in timm models list")

        model = timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=in_channels)
        encoder = model

        # SWIN produces channel-last: [B, H, W, C]
        if "swin" in name.lower():
            def swin_forward(x, model=model):
                x = model.forward_features(x)  # [B, H, W, C]
                if x.dim() == 4 and x.shape[-1] == out_ch:
                    x = x.permute(0, 3, 1, 2)  # → [B, C, H, W]
                return x
            encoder.forward = swin_forward

    else:
        raise ValueError(f"[feature_extractors] Unknown library type: {lib}")

    encoder = UnifiedBackbone(encoder, out_channels=out_ch)
    encoder.out_channels = out_ch
    return encoder

if __name__ == "__main__":
    _BACKBONES_YAML = "training/configs/backbones.yaml"
    backbones = load_backbones_config(_BACKBONES_YAML)

    DEVICE = 'mps'  # oppure "cuda"
    BATCH_SIZE = 1
    NUM_CHANNELS = 14
    TEST_SIZE = 224

    for entry in backbones:
        name = entry["name"]
        lib = entry.get("library", "")
        print(f"\n=== Testing backbone: {name} (library: {lib}) ===")
        try:
            enc = build_encoder_from_cfg(entry, in_channels=NUM_CHANNELS).to(DEVICE)
            print("  → out_channels from cfg:", getattr(enc, "out_channels", None))
            enc.eval()
            with torch.no_grad():
                dummy = torch.randn(BATCH_SIZE, NUM_CHANNELS, TEST_SIZE, TEST_SIZE).to(DEVICE)
                out = enc(dummy)
            if isinstance(out, list):
                print(f" → OK (multiscale), last shape: {out[-1].shape}, num_scales={len(out)}")
            else:
                print(f" → OK, output shape: {out.shape}")
        except Exception as e:
            print(f" !!! ERROR building or forward for {name}: {e}")