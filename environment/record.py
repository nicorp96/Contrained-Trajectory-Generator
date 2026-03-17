import copy
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import h5py

from mani_skil_ds_n import load_h5_data
import extended_peg_insertion

# from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


class ManiSkillTrajectoryDatasetRepro(Dataset):
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
        return len(self.episodes)

    def __getitem__(self, idx):
        eps = self.episodes[idx]
        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
        eps_len = len(trajectory["actions"])
        obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
        actions = trajectory["actions"]
        terminated = trajectory["terminated"]
        truncated = trajectory["truncated"]
        actions = common.to_tensor(actions, device=self.device)
        obs = common.index_dict_array(obs, idx, inplace=False)

        res = dict(
            obs=obs,
            env_states=trajectory["env_states"],
            actions=actions,
            terminated=terminated,
            truncated=truncated,
            seed=eps["episode_seed"],
        )
        # if self.rewards is not None:
        #     res.update(reward=selfrewards[idx])
        # if self.success is not None:
        #     res.update(success=self.success[idx])
        # if self.fail is not None:
        #     res.update(fail=self.fail[idx])
        return res


class AddTCPPoseAndDeltaPose(gym.ObservationWrapper):
    def __init__(self, env, time_key="time", delta_key="tcp_delta_pose"):
        super().__init__(env)
        self.prev_pose = None
        self.time_key = time_key
        self.delta_key = delta_key

    def reset(self, **kwargs):
        self.prev_pose = None
        return super().reset(**kwargs)

    def _get_tcp_pose_vec(self):
        tcp_pose = self.env.unwrapped.agent.tcp.pose
        p = np.asarray(tcp_pose.p, dtype=np.float32)
        q = np.asarray(tcp_pose.q, dtype=np.float32)
        return np.concatenate([p[0], q[0]], axis=0)

    def _delta_pose(self, prev_pose, cur_pose):
        p0, q0 = prev_pose[:3], prev_pose[3:]
        p1, q1 = cur_pose[:3], cur_pose[3:]

        dp = p1 - p0
        r0 = R.from_quat(q0)
        r1 = R.from_quat(q1)
        drot = (r1 * r0.inv()).as_rotvec().astype(np.float32)
        return np.concatenate([dp, drot], axis=0)

    def observation(self, obs):

        cur_pose = self._get_tcp_pose_vec()

        # time
        if hasattr(self.env.unwrapped, "scene") and hasattr(
            self.env.unwrapped.scene, "get_sim_time"
        ):
            t = float(self.env.unwrapped.scene.get_sim_time())
        else:
            t = float(getattr(self.env.unwrapped, "elapsed_steps", 0))

        # delta pose
        if self.prev_pose is None:
            delta_pose = np.zeros(6, dtype=np.float32)
        else:
            delta_pose = self._delta_pose(self.prev_pose, cur_pose)

        self.prev_pose = cur_pose.copy()

        # -------- GRIPPER STATE --------
        # Panda has two finger joints at the end of qpos
        qpos = self.env.unwrapped.agent.robot.get_qpos()
        finger_qpos = qpos[0, -2:]

        # normalize to [0,1]
        gripper = finger_qpos.mean().unsqueeze(0)

        # full action vector (7)
        delta_pose = torch.tensor(delta_pose, dtype=torch.float32)

        # -------- SAVE TO OBS --------
        obs["agent"][self.time_key] = torch.tensor([[t]], dtype=torch.float32)
        obs["agent"][self.delta_key] = torch.cat([delta_pose, gripper]).unsqueeze(0)

        return obs


class AddSimTimeToDictObs(gym.ObservationWrapper):
    def __init__(self, env, key="sim_time"):
        super().__init__(env)
        self.key = key

        # Update observation_space if it is a Dict space
        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces = dict(env.observation_space.spaces)
            spaces[self.key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        # Get sim time (most ManiSkill versions)
        if hasattr(self.env.unwrapped, "scene") and hasattr(
            self.env.unwrapped.scene, "get_sim_time"
        ):
            t = float(self.env.unwrapped.scene.get_sim_time())
        else:
            # fallback: step count (less ideal)
            t = float(getattr(self.env.unwrapped, "elapsed_steps", 0))
        # Add to dict as shape (1,) float32
        obs["agent"][self.key] = torch.tensor([t], dtype=torch.float32).unsqueeze(0)
        return obs


if __name__ == "__main__":

    env = gym.make(
        "PegInsertionSide-Extended",
        obs_mode="state_dict+rgb+depth",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reconfiguration_freq=1,
    )
    env = AddTCPPoseAndDeltaPose(
        env, time_key="time", delta_key="tcp_delta_pos"
    )  # <-- adds timestamp into obs

    env = RecordEpisode(
        env,
        output_dir="/mnt/data_nrp/dataset/mani_skill_data/demos/PegInsertionSide-DPT/motionplanning",
        save_trajectory=True,
        save_video=False,
        record_env_state=True,
    )

    DEMO_H5 = "/mnt/data_nrp/dataset/mani_skill_data/demos/PegInsertionSide-DPT/motionplanning/trajectory.h5"
    dataset = ManiSkillTrajectoryDatasetRepro(
        DEMO_H5, load_count=100, success_only=True, device="cpu"
    )
    for data in tqdm(dataset, total=len(dataset), desc="Replaying demos"):
        actions = data["actions"].numpy()
        states = data["env_states"]  # list/seq of dicts
        seed = data["seed"]
        sd0 = common.to_numpy(states)  # initial state dict
        # Optional: set exact sim state
        obs, info = env.reset(seed=int(seed))
        if states is not None:
            env.unwrapped.set_state_dict(common.to_numpy(states))
            if hasattr(env.unwrapped, "get_obs"):
                obs = env.unwrapped.get_obs()
        # recompute obs from the loaded state (reset obs is stale)
        obs = env.unwrapped.get_obs() if hasattr(env.unwrapped, "get_obs") else obs
        done = False
        for action in actions:
            obs, rew, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            # env.render_human()
    env.close()
