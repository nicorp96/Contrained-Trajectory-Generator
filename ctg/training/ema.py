import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of a model's weights with warm-up
    decay = 1 - (1 + step / inv_gamma) ** -power, clipped to [min_decay, max_decay]."""

    def __init__(self, model: nn.Module, update_after_step: int = 0, inv_gamma: float = 1.0, power: float = 0.75,
                 min_decay: float = 0.0, max_decay: float = 0.9999):
        self.model = model.eval().requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma, self.power = inv_gamma, power
        self.min_decay, self.max_decay = min_decay, max_decay
        self.optimization_step = 0

    def decay(self, step: int) -> float:
        step = max(0, step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return 0.0 if value <= 0.0 else max(self.min_decay, min(value, self.max_decay))

    @torch.no_grad()
    def step(self, model: nn.Module):
        decay = self.decay(self.optimization_step)
        source = model.state_dict()
        for name, ema_value in self.model.state_dict().items():
            value = source[name].to(ema_value.dtype)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(value, alpha=1 - decay)
            else:  # integer buffers (e.g. counters) are copied
                ema_value.copy_(value)
        self.optimization_step += 1
