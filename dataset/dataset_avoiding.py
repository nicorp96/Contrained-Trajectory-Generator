import os, pickle
from typing import Dict, List, Tuple
import numpy as np
import torch

from dataset.base_dataset import BaseTrajectoryDS
from dataset.utils import get_max_min, get_mean_std
from models.utils.normalizer import DictNormalizer


class AvoidingTrajDataset(BaseTrajectoryDS):
    """
    Dataset for d3il avoiding trajectories.

    Each sample is a fixed-length window of length H = config["horizon"]:

        {
            "states":      [H, state_dim]   (padded with zeros if needed)
            "actions":     [H, action_dim]  (padded with zeros if needed)
            "environment": {"goal": [goal_dim]}   # here: final desired pos of episode
        }
    """

    def __init__(self, config: Dict, val: bool = False):
        super().__init__(config, val)
        self.horizon = config["horizon"]  # H
        self.max_path_length = config["max_path_length"]
        # self.state_act_shapes = config["state_action_shapes"]
        self.state_shapes = config["state_shapes"]
        self.action_shapes = config["actions_shapes"]
        self.environment_shapes = config["environment_shapes"]
        # Per-episode tensors (variable length)
        self._ep_states: List[torch.Tensor] = []  # each [T, state_dim]
        self._ep_actions: List[torch.Tensor] = []  # each [T, action_dim]
        self._ep_goals: List[torch.Tensor] = []  # each [goal_dim]
        self._path_lengths: List[int] = []
        action = []
        state = []
        goal_lst = []
        # self.dict_norm = DictNormalizer(self.state_act_shapes)
        self.states_norm = DictNormalizer(self.state_shapes)
        self.actions_norm = DictNormalizer(self.action_shapes)
        self.environment_norm = DictNormalizer(self.environment_shapes)
        # Build episodes
        state_files = sorted(os.listdir(self.root))
        for file in state_files:
            file_path = os.path.join(self.root, file)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "rb") as f:
                env_state = pickle.load(f)

            # ---- extract raw positions ----
            # shape: [T_raw, 2]
            robot_des_pos = np.asarray(env_state["robot"]["des_c_pos"])[:, :2]
            robot_c_pos = np.asarray(env_state["robot"]["c_pos"])[:, :2]

            # state_t = [des_x, des_y, cur_x, cur_y]
            state_seq = np.concatenate([robot_des_pos, robot_c_pos], axis=-1)
            state.append(state_seq)
            # actions: vel over desired position
            # len(vel) = T_raw - 1
            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
            action.append(vel_state)

            T = len(vel_state)
            if T <= 0:
                continue

            # truncate if too long
            if T > self.max_path_length:
                T = self.max_path_length
                # states are defined at action times: take first T states
                state_seq = state_seq[:T]
                vel_state = vel_state[:T]
            else:
                # align states with actions: drop last state
                state_seq = state_seq[:-1]

            assert state_seq.shape[0] == vel_state.shape[0]
            T = state_seq.shape[0]

            # convert to torch
            states_t = torch.from_numpy(state_seq).float()  # [T, state_dim]
            actions_t = torch.from_numpy(vel_state).float()  # [T, action_dim]

            # goal = final desired position of episode (before any truncation)
            goal = torch.from_numpy(robot_des_pos[-1]).float()  # [2]
            goal_lst.append(goal)
            self._ep_states.append(states_t)
            self._ep_actions.append(actions_t)
            self._ep_goals.append(goal)
            self._path_lengths.append(T)

        if len(self._ep_states) == 0:
            raise RuntimeError(f"No valid trajectories found in {self.root}")

        # Build window indices (episode_id, start_t)
        self._indices: List[Tuple[int, int]] = self._make_indices(
            self._path_lengths, self.horizon
        )
        self.stats = self.compute_stats(state, action, goal_lst)
        # Cache dims
        self.state_dim = self._ep_states[0].shape[1]
        self.action_dim = self._ep_actions[0].shape[1]
        self.goal_dim = self._ep_goals[0].shape[0]

    @staticmethod
    def compute_stats(state, action, goal):
        stats = {}
        max_st, min_st = get_max_min(state)
        max_act, min_act = get_max_min(action)
        max_goal, min_goal = get_max_min(goal)

        mean_st, std_st = get_mean_std(state)
        mean_act, std_act = get_mean_std(action)
        mean_goal, std_goal = get_mean_std(goal)
        stats.update(
            {
                "desired_pos": {
                    "max": max_st[:2],
                    "min": min_st[:2],
                    "mean": mean_st[:2],
                    "std": std_st[:2],
                },
                "current_pos": {
                    "max": max_st[2:],
                    "min": min_st[2:],
                    "mean": mean_st[2:],
                    "std": std_st[2:],
                },
            }
        )
        stats.update(
            {
                "vel": {
                    "max": max_act,
                    "min": min_act,
                    "mean": mean_act,
                    "std": std_act,
                }
            }
        )
        stats.update(
            {
                "goal": {
                    "max": max_goal,
                    "min": min_goal,
                    "mean": mean_goal,
                    "std": std_goal,
                }
            }
        )

        return stats

    @staticmethod
    def _make_indices(path_lengths: List[int], horizon: int) -> List[Tuple[int, int]]:
        """
        Build list of (episode_id, start_t) for windows of length horizon.
        If an episode is shorter than horizon, we still create one window starting at 0
        and will pad in __getitem__.
        """
        indices: List[Tuple[int, int]] = []
        for ep, T in enumerate(path_lengths):
            if T >= horizon:
                max_start = T - horizon + 1
                for s in range(max_start):
                    indices.append((ep, s))
            else:
                # too short -> one window starting at 0, padded later
                indices.append((ep, 0))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        ep_id, start = self._indices[idx]

        states_ep = self._ep_states[ep_id]  # [T, state_dim]
        actions_ep = self._ep_actions[ep_id]  # [T, action_dim]
        T = states_ep.shape[0]
        H = self.horizon

        # effective end of slice
        end = min(start + H, T)

        # allocate padded window
        states = torch.zeros(H, self.state_dim, dtype=torch.float32)
        actions = torch.zeros(H, self.action_dim, dtype=torch.float32)

        seg_len = end - start
        states[:seg_len] = states_ep[start:end]
        actions[:seg_len] = actions_ep[start:end]
        states_dict = {
            "desired_pos": states[:, :2],
            "current_pos": states[:, 2:],
        }
        actions_dict = {
            "vel": actions,
        }
        goal = self._ep_goals[ep_id]  # [goal_dim]

        sample = {
            "states": states_dict,  # [H, state_dim]
            "actions": actions_dict,  # [H, action_dim]
            "environment": {"goal": goal},
            # optionally also expose seg_len if you want a mask later
            # "length": seg_len,
        }
        return sample

    def get_normalizer(self):
        stats = self.stats
        self.states_norm.set_stats_from_dict(stats)
        self.actions_norm.set_stats_from_dict(stats)
        self.environment_norm.set_stats_from_dict(stats)
        # self.dict_norm.set_stats_from_dict(stats)
        return self.states_norm, self.actions_norm, self.environment_norm


# if __name__ == "__main__":
#     # quick smoke test
#     root = "/home/nrodriguez/Documents/DProjekt/research_t_opt/synth_trajs"
#     train_loader = make_loader(root, "train", batch_size=8, shuffle=True, num_workers=0)
#     val_loader = make_loader(root, "val", batch_size=8, shuffle=False, num_workers=0)
#     batch = next(iter(train_loader))
#     print(
#         {
#             k: (
#                 {kk: vv.shape for kk, vv in batch[k].items()}
#                 if isinstance(batch[k], dict)
#                 else type(batch[k])
#             )
#             for k in batch
#         }
#     )
#     batch = next(iter(val_loader))
#     print(
#         {
#             k: (
#                 {kk: vv.shape for kk, vv in batch[k].items()}
#                 if isinstance(batch[k], dict)
#                 else type(batch[k])
#             )
#             for k in batch
#         }
#     )
