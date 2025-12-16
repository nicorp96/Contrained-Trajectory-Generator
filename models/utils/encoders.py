import torch.nn as nn
import torchvision.models as models
from typing import Dict


class DictEncoder(nn.Module):
    def __init__(self, param_shapes: dict):
        super(DictEncoder, self).__init__()
        self.encoders = nn.ModuleDict()

        for name in param_shapes.keys():
            type = param_shapes[name]["type"]
            net = None
            if type == "resnet18":
                net = models.resnet18(pretrained=param_shapes.get("pretrained", True))
                modules = list(net.children())[
                    :-1
                ]  # remove the classification head (fc layer)
                net = nn.Sequential(*modules, nn.Flatten())  # flatten output to [B, D]
                output_dim = param_shapes.get(
                    "output_dim", 512
                )  # resnet18 final feature dim

            elif type == "resnet50":
                net = models.resnet50(pretrained=param_shapes.get("pretrained", True))
                modules = list(net.children())[:-1]
                net = nn.Sequential(*modules, nn.Flatten(), nn.Linear())
                output_dim = param_shapes.get(
                    "output_dim", 2048
                )  # resnet50 final feature dim
            elif type == "gru":
                net = nn.Sequential(
                    nn.Linear(
                        param_shapes[name]["d_obs"], param_shapes[name]["d_model"]
                    ),
                    nn.GRU(
                        param_shapes[name]["d_model"],
                        param_shapes[name]["d_model"] // 2,
                        num_layers=param_shapes[name]["n_layers"],
                        batch_first=True,
                        dropout=param_shapes[name]["dropout"],
                        bidirectional=True,
                    ),
                    nn.LayerNorm(param_shapes[name]["d_model"]),
                )
            elif type == "cnn":
                net = nn.Sequential(
                    nn.Conv1d(
                        in_channels=param_shapes[name]["in_channels"],
                        out_channels=param_shapes[name]["out_channels"],
                        kernel_size=param_shapes[name]["kernel_size"],
                        stride=param_shapes[name]["stride"],
                        padding=param_shapes[name]["padding"],
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            elif type == "linear":
                net = nn.Sequential(
                    nn.Linear(
                        param_shapes[name]["d_obs"], param_shapes[name]["d_model"]
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        param_shapes[name]["d_model"], param_shapes[name]["d_model"]
                    ),
                )
            elif type == "identity":
                net = nn.Identity()
            else:
                raise ValueError(f"Unknown encoder type: {type}")

            self.encoders[name] = net

    # def set_to_device(self, device: torch.device):
    #     for name, net in self.encoders.items():
    #         self.encoders[name] = net.to(device)

    def forward(self, observations: Dict):
        return {
            name: self.encoders[name](value) for name, value in observations.items()
        }
