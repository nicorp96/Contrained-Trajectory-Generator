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
