import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMA:
    """
    Exponential Moving Average (EMA) module.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model.eval()
        self.model.requires_grad_(False)
        self.update_after_step = config["update_after_step"]
        self.inv_gamma = config["inv_gamma"]
        self.power = config["power"]
        self.min_value = config["min_value"]
        self.max_value = config["max_value"]

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if value <= 0.0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        model_sd = new_model.state_dict()
        ema_sd = self.model.state_dict()

        for name, ema_param in ema_sd.items():
            if name not in model_sd:
                continue
            param = model_sd[name]
            if not torch.is_floating_point(ema_param):
                # non-float buffers: just copy
                ema_param.copy_(param)
                continue

            if ema_param.shape != param.shape:
                ema_param.copy_(param.to(dtype=ema_param.dtype))
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        self.model.load_state_dict(ema_sd)
        self.optimization_step += 1

    # def step(self, new_model):
    #     self.decay = self.get_decay(self.optimization_step)
    #     for module, ema_module in zip(new_model.modules(), self.model.modules()):
    #         for param, ema_param in zip(
    #             module.parameters(recurse=False), ema_module.parameters(recurse=False)
    #         ):
    #             if isinstance(param, dict):
    #                 raise RuntimeError("Dict parameter not supported")
    #
    #             if isinstance(module, _BatchNorm):
    #                 ema_param.copy_(param.to(dtype=ema_param.dtype).data)
    #             elif not param.requires_grad:
    #                 ema_param.copy_(param.to(dtype=ema_param.dtype).data)
    #             else:
    #                 ema_param.mul_(self.decay)
    #                 ema_param.add_(
    #                     param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
    #                 )
    #
    #     self.optimization_step += 1
