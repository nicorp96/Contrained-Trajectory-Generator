import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import AutoModel, AutoImageProcessor


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.preprocess = tvm.ResNet18_Weights.DEFAULT.transforms()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()
        self.backbone = model
        self.output_dim = 512

    def forward(self, x):
        x = self.preprocess(x)  # [B, C, H, W]
        return self.backbone(x)  # [B, 512]


class HFVisionEncoder(nn.Module):
    def __init__(self, model_name, pool="cls"):
        super().__init__()
        self.model_name = model_name
        self.pool = pool
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out_dim = self._infer_dim(dummy)
        self.output_dim = out_dim

    def _infer_dim(self, x):
        feat = self.forward(x)
        return feat.shape[-1]

    def forward(self, x):
        dev = next(self.backbone.parameters()).device
        x = x.to(dev)

        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        out = self.backbone(**inputs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output

        tokens = out.last_hidden_state
        if self.pool == "cls":
            return tokens[:, 0]
        elif self.pool == "mean":
            return tokens.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool={self.pool}")


def freeze_all_parameters(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_resnet_last_n(encoder, n=1):
    # assumes encoder.encoder is a torchvision resnet
    layers = ["layer1", "layer2", "layer3", "layer4"]
    for layer_name in layers[-n:]:
        layer = getattr(encoder.encoder, layer_name, None)
        if layer is not None:
            for p in layer.parameters():
                p.requires_grad_(True)


def unfreeze_hf_last_n(encoder, n=1):
    # common HF transformer layout
    backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder.encoder

    # try common transformer block containers
    candidate_paths = [
        ("encoder", "layer"),  # BERT/ViT-like
        ("encoder", "layers"),  # some variants
        ("layers",),  # some custom models
        ("layer",),  # fallback
    ]

    blocks = None
    for path in candidate_paths:
        obj = backbone
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            blocks = obj
            break

    if blocks is None:
        # fallback: unfreeze everything if structure unknown
        for p in backbone.parameters():
            p.requires_grad_(True)
        return

    for block in list(blocks)[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)
