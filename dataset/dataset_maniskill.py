from einops import rearrange
import h5py
import json
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from common.resampling_algo import (
    ResamplingBase,
    build_schema_from_layout,
    prepare_strictly_increasing_x,
)
from models.utils.normalizer import DictNormalizer
from einops import repeat
from common.get_class import get_class_dict
from global_parameters import ConfigGlobalP

cfg_gp = ConfigGlobalP()


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class BaseDataset(Dataset):
    def __init__(self, config, dtype=torch.float32):
        self.config = config
        self.h5_file = None
        self.stats_path = None
        self.dtype = dtype
        self.__load_file__(config["path"])
        self.__load_paths__(config["path_stats"])
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = None
        self.success, self.fail, self.rewards = None, None, None
        for eps_id in tqdm(range(len(self.episodes))):
            eps = self.episodes[eps_id]
            assert (
                "success" in eps
            ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
            if not eps["success"]:
                self.episodes.pop(eps_id)
                continue

    def __len__(self):
        return len(self.episodes)

    def __load_file__(self, path):
        path_root_data = None
        if not os.path.exists(path):
            path_root_data = os.path.join(cfg_gp.DATA_DIR, path)
            if not os.path.exists(path_root_data):
                raise FileNotFoundError(f"{path} does not exist")
        self.h5_file = h5py.File(path_root_data, "r")
        json_path = path_root_data.replace(".h5", ".json")
        self.json_data = load_json(json_path)

    def __load_paths__(self, path):
        path_root_data = None
        if not os.path.exists(path):
            path_root_data = cfg_gp.DATA_DIR.joinpath(path)
            if not path_root_data.exists():
                raise FileNotFoundError(f"{path} does not exist")
        self.stats_path = path_root_data


class TrajectoryKDatasetHDF5(BaseDataset):
    def __init__(self, config, dtype=torch.float32):
        super().__init__(config, dtype)
        self.state_shapes = config["state_shapes"]
        self.goal_shapes = config["goal_shapes"]
        self.expand_len = config["expand_goal"]
        self.obs_shapes = config.get("eval_shapes")
        self.dict_norm_states = DictNormalizer(self.state_shapes)
        self.dict_norm_goals = DictNormalizer(self.goal_shapes)
        self.dict_norm_contex = DictNormalizer(self.obs_shapes)
        self.sequence_len_H = self.config.get("sequence_hist", 6)
        # build split slices and angle column indices in the concatenated D
        self._slices = {}
        self._angle_cols = []
        offset = 0
        resampling_states = list(self.state_shapes.keys())
        # if "delta_pos" in resampling_states:
        #     resampling_states = (
        #         list("position") + resampling_states
        #     )  # list(self.obs_shapes.keys())
        for k in resampling_states:
            if k in self.state_shapes:
                d = int(self.state_shapes[k]["shape"])
            elif k in self.obs_shapes:
                d = int(self.obs_shapes[k]["shape"])
            self._slices[k] = slice(offset, offset + d)
            if "rad" in k or "deg" in k:
                # all channels of this key are angles; customize if only some are angles
                self._angle_cols.extend(list(range(offset, offset + d)))
            offset += d

        self.resampling_clss: ResamplingBase = (
            get_class_dict(config["resampling_algo"])
            if "resampling_algo" in config
            else None
        )
        self.D_total = offset
        self.schema = build_schema_from_layout(
            self.state_shapes, interp_kind=self.resampling_clss.interp_kind
        )
        self.start_fixed_idx = None

    def __len__(self):
        return len(self.h5_file[list(self.state_shapes.keys())[0]])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def set_fixed_idx(self, fixed_idx):
        self.start_fixed_idx = fixed_idx

    def get_stats_from_file(self):
        stats_json_path = self.stats_path
        with open(stats_json_path, "r") as f:
            json_stats_dict_r = json.load(f)
        self.dict_norm_states.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_goals.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_contex.set_stats_from_json(json_stats_dict_r)
        return self.dict_norm_states, self.dict_norm_goals, self.dict_norm_contex

    def __getitem__(self, idx):
        hdf5_data = self.h5_file
        dtype = self.dtype
        seq_len_H = self.sequence_len_H
        resampling_states = list(self.state_shapes.keys())
        start_fixed_idx = 250  # self.start_fixed_idx
        # if "delta_pos" in resampling_states:
        #     resampling_states = resampling_states + list(self.obs_shapes.keys())
        sample_state = {}
        sample_goal = {}
        extra_shapes_dict = {}
        state_keys = self.state_shapes.keys()
        goal_keys = self.goal_shapes.keys()
        obs_shapes = self.obs_shapes
        # --- load time (M,)
        t_np = hdf5_data["time"][str(idx)][...]
        t = np.squeeze(t_np, 1)

        start_idx = (
            np.random.randint(0, (t.shape[0] - 2))
            if start_fixed_idx is None
            else start_fixed_idx
        )
        seq_len_L = t.shape[0] - start_idx
        # t = torch.as_tensor(t_np, dtype=self.dtype).squeeze(1)
        # start_idx = torch.randint(0, (t.size(0) - 2), ())
        parts = []
        for k in resampling_states:
            # if "delta_pos" in k:
            #     continue
            data = hdf5_data[k][str(idx)][...]  # (M, d_k)
            parts.append(data[start_idx:])
        X = np.concatenate(parts, axis=1)
        t_in = t[start_idx:]
        t_in, X = prepare_strictly_increasing_x(t_in, X)
        # t_new, Y, _dt, T = self.resampling_clss.resample(t_in, states=X, dt=None)
        out_dict = self.resampling_clss.resample(
            t_in, states=X, schema=self.schema, dt=None
        )
        t_new = torch.from_numpy(out_dict["t_new"]).to(dtype)
        Y = torch.from_numpy(out_dict["X"]).to(dtype)
        _dt = torch.tensor(out_dict["dt"]).to(dtype)
        for k, sl in self._slices.items():
            if k in state_keys:
                # if k == "delta_pos":
                #     continue
                sample_state[k] = Y[:, sl]  # (N, d_k)
                # if k == "position":
                #     sample_state[k] = Y[:, sl]  # - Y[0, :3]
            elif k in obs_shapes.keys():
                if k == "position":
                    pos = Y[:, sl]
                    dpos = torch.zeros_like(pos, dtype=dtype)
                    dpos[1:] = torch.diff(pos, dim=0)
                    assert torch.allclose(dpos[2, 0], pos[2, 0] - pos[1, 0])
                    sample_state["delta_pos"] = dpos

        for key in goal_keys:
            data = hdf5_data[key][str(idx)][...]
            if "deg" in key:
                sample_goal[key] = torch.deg2rad(torch.tensor(data, dtype=dtype))
            if "rad" in key:
                sample_goal[key] = torch.rad2deg(torch.tensor(data, dtype=dtype))[:1, :]
            elif "km" in key:
                sample_goal[key] = torch.tensor(data, dtype=dtype) * cfg_gp.KM_TO_M
                # x, y, z in NED --> z = -alt
                sample_goal[key][2] = sample_goal[key][2] * -1.0
            else:
                sample_goal[key] = torch.tensor(data, dtype=dtype)  # [210:, :]
            if data.ndim < 2 and self.expand_len > 0:
                sample_goal[key] = repeat(
                    sample_goal[key], "f -> s f", s=self.expand_len
                )
        if obs_shapes is not None:
            for key in obs_shapes:
                # # Assume data stored as [num_samples, ...] shape in HDF5
                # if key == "position":
                #     extra_shapes_dict[key] = Y[:, :3]
                # else:
                #     data = hdf5_data[key][str(idx)][...]
                #     extra_shapes_dict[key] = torch.tensor(data[start_idx:], dtype=dtype)
                if start_idx > seq_len_H:
                    value_history = hdf5_data[key][str(idx)][
                        (start_idx - seq_len_H) : start_idx, :
                    ]
                elif start_idx == 0:
                    data_cache = hdf5_data[key][str(idx)][...][0]
                    value_history = np.tile(data_cache, (seq_len_H, 1))
                else:
                    diff_idx = seq_len_H - start_idx
                    data_cache = hdf5_data[key][str(idx)][:start_idx, :]
                    value_history = np.zeros((seq_len_H, data_cache.shape[1]))

                    # value_history[:start_idx, :] = data_cache
                    value_history[:diff_idx, :] = np.tile(
                        data_cache[0, :], (diff_idx, 1)
                    )
                    value_history[diff_idx:, :] = data_cache

                # data = hdf5_data[key][str(idx)][...]
                extra_shapes_dict[key] = torch.tensor(value_history, dtype=dtype)
            extra_shapes_dict["mask"] = torch.zeros((seq_len_H, 1), dtype=dtype)
            extra_shapes_dict["seq_len_L"] = seq_len_L - 1
            # extra_shapes_dict["t_new"] = t_new
            # extra_shapes_dict["dt"] = _dt
            return sample_state, sample_goal, extra_shapes_dict
        return sample_state, sample_goal


class TrajectoryPadDatasetHDF5(BaseDataset):
    def __init__(self, config, dtype=torch.float32):
        super(TrajectoryPadDatasetHDF5, self).__init__(config, dtype)
        self.state_shapes = config["state_shapes"]
        self.goal_shapes = config["goal_shapes"]
        self.expand_len = config["expand_goal"]
        self.obs_shapes = config.get("obs_shapes")
        self.dict_norm_states = DictNormalizer(self.state_shapes)
        self.dict_norm_goals = DictNormalizer(self.goal_shapes)
        self.dict_norm_contex = DictNormalizer(self.obs_shapes)
        self.sequence_len = self.config.get("sequence_len", 256)
        self.sequence_len_H = self.config.get("sequence_hist", 6)
        self.dict_norm_contex = DictNormalizer(self.obs_shapes)

    @staticmethod
    def padding_with_abs_tail(
        value: torch.Tensor, fixed_length: int, key: str
    ) -> torch.Tensor:
        L, D = value.shape

        if key == "action":
            value_ret = torch.zeros((fixed_length, D))
        else:
            value_ret = torch.zeros((fixed_length, D)) + value[-1, :]
        value_ret[:L, :] = value
        return value_ret

    def __getitem__(self, idx):
        eps = self.episodes[idx]
        hdf5_data = self.h5_file
        dtype = self.dtype
        sequence_len = self.sequence_len
        seq_len_H = self.sequence_len_H
        state_dict = {}
        obs_dict = {}
        context_shape_dict = {}
        state_keys = self.state_shapes.keys()
        obs_keys = self.goal_shapes.keys()
        obs_shapes = self.obs_shapes
        trajectory = hdf5_data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
        eps_len = len(trajectory["actions"])
        obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
        start_idx = np.random.randint(0, (eps_len - 2))
        mask = torch.zeros((sequence_len), dtype=torch.bool)  # , dtype=torch.bool)
        seq_len_L = eps_len - start_idx
        mask[:seq_len_L] = 1.0
        for key in state_keys:
            data = None
            for obs_key in obs.keys():
                if key in obs[obs_key].keys():
                    data = obs[obs_key][key]
                    break
            if key in trajectory:
                data = trajectory[key]
            elif data is None:
                print(f"Key {key} in state keys not found in DS")
                continue
            state_dict[key] = self.padding_with_abs_tail(
                torch.tensor(data, dtype=dtype), sequence_len, key
            )
        for key in obs_keys:
            data = None
            for obs_key in obs.keys():
                if key in obs[obs_key].keys():
                    data = obs[obs_key][key]
                    break
            if data is None:
                print(f"Key {key} in obs keys not found in DS")
                continue
            obs_dict[key] = torch.tensor(data[-1, :], dtype=dtype)

        for key in obs_shapes:
            history_data = None
            for obs_key in obs.keys():
                if key in obs[obs_key].keys() and obs_key != "sensor_param":
                    if "camera" in key:
                        history_data = obs[obs_key][key][obs_shapes[key]["type"]]
                        history_data = history_data.transpose(0, 3, 1, 2)
                        break
                    history_data = obs[obs_key][key]
                    break
            if history_data is None:
                print(f"Key {key} in context keys not found in DS")
                continue

            if start_idx > seq_len_H:
                value_history = history_data[(start_idx - seq_len_H) : start_idx, :]
            # elif start_idx == 0:
            #     data_cache = history_data[0]
            #     value_history = np.tile(data_cache, (seq_len_H, 1))
            else:
                diff_idx = seq_len_H - start_idx
                if start_idx == 0:
                    data_cache = np.expand_dims(history_data[0, :], axis=0)
                else:
                    data_cache = history_data[:start_idx, :]
                if history_data.ndim > 2:
                    data_tensor = repeat(
                        torch.from_numpy(data_cache)[-1],
                        "C H W -> L C H W",
                        L=diff_idx,
                    )
                    # value_history = torch.cat(
                    #     [torch.from_numpy(data_cache), data_tensor], dim=0
                    # ).numpy()
                else:
                    data_tensor = repeat(
                        torch.from_numpy(data_cache)[-1],
                        "D -> L D",
                        L=diff_idx,
                    )

                value_history = (
                    torch.cat(
                        [torch.from_numpy(data_cache), data_tensor], dim=0
                    ).numpy()
                    if start_idx > 0
                    else data_tensor.numpy()
                )

            context_shape_dict[key] = torch.tensor(value_history, dtype=dtype)
        context_shape_dict["mask"] = mask
        context_shape_dict["seq_len_L"] = seq_len_L - 1
        return state_dict, obs_dict, context_shape_dict

    def get_stats_from_file(self):
        stats_json_path = self.stats_path
        with open(stats_json_path, "r") as f:
            json_stats_dict_r = json.load(f)
        self.dict_norm_states.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_goals.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_contex.set_stats_from_json(json_stats_dict_r)
        return self.dict_norm_states, self.dict_norm_goals, self.dict_norm_contex
