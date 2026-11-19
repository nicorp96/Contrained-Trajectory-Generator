from collections import deque
from typing import Dict

import torch


class ObservationHistory:
    """Rolling window of the last `length` observations per key. The first observation
    fills the whole window, matching how the dataset pads the start of an episode."""

    def __init__(self, length: int):
        self.length = int(length)
        self.buffers: Dict[str, deque] = {}

    def reset(self):
        self.buffers = {}

    def update(self, observation: Dict[str, torch.Tensor]):
        for key, value in observation.items():
            value = value.detach().clone()
            if value.shape[0] == 1:  # drop a batch dimension of size 1
                value = value.squeeze(0)
            if key not in self.buffers:
                self.buffers[key] = deque([value.clone() for _ in range(self.length)], maxlen=self.length)
            else:
                self.buffers[key].append(value)

    def stacked(self) -> Dict[str, torch.Tensor]:
        """{key: [1, length, ...]}"""
        return {k: torch.stack(list(v)).unsqueeze(0) for k, v in self.buffers.items()}
