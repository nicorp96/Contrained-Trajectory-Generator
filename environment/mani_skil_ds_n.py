from typing import Union

import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(
        self, dataset_file: str, load_count=-1, success_only: bool = False, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = None
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only:
                assert (
                    "success" in eps
                ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps["success"]:
                    self.episodes.pop(eps_id)
                    continue

    def __len__(self):
        return  len(self.episodes)

    def __getitem__(self, idx):
        eps = self.episodes[idx]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
        eps_len = len(trajectory["actions"])
        obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
        actions =trajectory["actions"]
        terminated = trajectory["terminated"]
        truncated = trajectory["truncated"]
        actions = common.to_tensor(actions, device=self.device)
        #obs = common.index_dict_array(obs, idx, inplace=False)

        res = dict(
            obs=obs,
            actions=actions,
        )
        # if self.rewards is not None:
        #     res.update(reward=selfrewards[idx])
        # if self.success is not None:
        #     res.update(success=self.success[idx])
        # if self.fail is not None:
        #     res.update(fail=self.fail[idx])
        return res

if __name__ == "__main__":
    DEMO_H5 = "/mnt/data_nrp/dataset/mani_skill_data/demos/PegInsertionSide-ExtendedTIME/motionplanning/trajectory.h5"
    dataset = ManiSkillTrajectoryDataset(DEMO_H5, load_count=-1, success_only=True, device=None)
    for data in dataset:
        print(type(data["actions"]))



