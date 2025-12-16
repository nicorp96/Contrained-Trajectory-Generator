from typing import Dict
from torch.utils.data import Dataset


class BaseTrajectoryDS(Dataset):
    """ """

    def __init__(
        self,
        config: Dict,
        val: bool = False,
    ):
        super().__init__()
        self.config = config
        self.root = config["root_dir"]
        self.split = "val" if val else "train"
        self.horizon = config["horizon"]

    def get_stats_from_file(self):
        return NotImplementedError
