from collections import deque
from einops import repeat, rearrange
import torch
import time


from common.utils import split_state_tensor
from common.inpainting import apply_inpainting, make_inpainting_mask
from models.policies.base_policy import Policy


class TensorHistory:
    def __init__(self, H):
        self.H = H
        self.history = {}

    def update(self, current_data: dict):
        for key, value in current_data.items():
            value = value.clone()

            # remove batch dimension if it is always 1
            if value.shape[0] == 1:
                value = value.squeeze(0)

            if key not in self.history:
                self.history[key] = deque(
                    [value.clone() for _ in range(self.H)],
                    maxlen=self.H,
                )

        for key, q in self.history.items():
            if key in current_data:
                value = current_data[key].clone()
                if value.shape[0] == 1:
                    value = value.squeeze(0)
                q.append(value)
            else:
                q.append(q[-1].clone())

    def get_stacked(self):
        return {
            k: torch.stack(list(v), dim=0).unsqueeze(0) for k, v in self.history.items()
        }


class DiffPolicy(Policy):
    def __init__(self, config, device, only_actions=True, **kwargs):
        super().__init__(config, device, only_actions, **kwargs)
        self.action_key = (
            kwargs["action_key"] if "action_key" in kwargs.keys() else "actions"
        )
        self.action_horizon = (
            kwargs["action_horizon"] if "action_horizon" in kwargs.keys() else 1
        )
        self.contex_history = TensorHistory(self.config["dataset"]["sequence_hist"])

    def get_dict_from_obs(self, obs, key):
        data = None
        if key in obs:
            data = obs[key]
        else:
            for obs_key in obs.keys():
                if obs_key == "sensor_param":
                    continue

                if key in obs[obs_key].keys():
                    data = obs[obs_key][key]
                    break
        if data is None:
            print(f"Key {key} not found in obs")
        return data

    def get_states_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["state_shapes"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            if key == self.action_key:
                data = self.get_dict_from_obs(
                    obs, key
                )  # [:, : shapes_dict[key]["shape"]]
                # data[:, -1] = 1.0
            data_zeros = repeat(
                data, "B D -> B L D", L=self.sequence_len
            )  # torch.zeros((1, self.sequence_len, shapes_dict[key]["shape"]))
            # data_zeros[:, -1, :] = data
            out[key] = data_zeros
        return out

    def get_goal_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["goal_shapes"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            out[key] = data + 0.1
        return out

    def get_context_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["obs_shapes"]
        sequence_hist = self.config["dataset"]["sequence_hist"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            if key == self.action_key:
                data = self.get_dict_from_obs(obs, key)
            if "camera" in key:
                data = data["rgb"]
                data = rearrange(data, "B H W C -> B C H W")
                # data_zeros = repeat(data, "B C H W -> B L C H W", L=sequence_hist)
                # data_zeros[:, -1, :, :, :] = data
            # else:
            # data_zeros = torch.zeros((1, sequence_hist, shapes_dict[key]["shape"]))
            # data_zeros[:, -1, :] = data
            out[key] = data  # _zeros
        return out

    def get_states_goal_extra_from_obs(self, obs):
        states_dict = self.get_states_from_obs(obs)
        goals_dict = self.get_goal_from_obs(obs)
        context = self.get_context_from_obs(obs)
        self.contex_history.update(context)
        return states_dict, goals_dict

    @torch.inference_mode()
    def conditional_sample(self, states_norm_dict, goals_norm, history_dict_nm, device):
        model = self.ema_model if self.ema_model is not None else self.model
        inpt = self.inpainting
        guidance_algo = self.guidance_algo
        scheduler = self.noise_scheduler
        trj = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        cond = torch.cat(
            [goals_norm[key] for key in goals_norm.keys()],
            dim=1,
        ).to(device)
        trajectory_history = torch.cat(
            [history_dict_nm[key].to(device) for key in history_dict_nm.keys()],
            dim=2,
        )
        mask_inp = make_inpainting_mask(
            trajectory=trj,
            start_indices=self.start_indices,
            goal_indices=self.goal_indices,
        )

        trajectory_n = torch.randn(size=trj.shape, dtype=trj.dtype, device=trj.device)

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )
        if inpt:
            trajectory_n = apply_inpainting(trajectory_n, mask_inp, trj, noise=False)
        for k in scheduler.timesteps:
            trajectory_n = guidance_algo(
                input_noised=trajectory_n,
                k=k,
                cond=cond,
                scheduler=scheduler,
                model=model,
                state_hist=trajectory_history,
            )
            if inpt:
                trajectory_n = apply_inpainting(
                    trajectory_n, mask_inp, trj, noise=False
                )
        return trajectory_n

    def __call__(self, obs):
        states_dict, goals_dict = self.get_states_goal_extra_from_obs(obs)
        device = self.device
        ctx_hist = self.contex_history.get_stacked()
        # Normalize
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        history_dict_nm = self.normalizer_obs(ctx_hist)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out_trj = self.conditional_sample(
            states_norm_dict, goals_norm, history_dict_nm, device=device
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"conditional_sample: {(t1 - t0) * 1000:.3f} ms")

        output_dict = split_state_tensor(
            out_trj, self.config["dataset"]["state_shapes"]
        )
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        if self.only_actions:
            output_unnormalized = output_unnormalized[self.action_key][
                :, : self.action_horizon, :
            ]
        return output_unnormalized


if __name__ == "__main__":
    config = {
        "path_ckp": "/mnt/data_nrp/research_t_opt/logs/GCDiTDiff/20251216-111751/logs/checkpoint_1195.pth"
    }
    device = torch.device("cuda:0")
    policy = DiffPolicy(config, device)
    states = {
        "desired_pos": torch.ones((1, 16, 2)),
        "current_pos": torch.ones((1, 16, 2)),
    }
    actions = {"vel": torch.ones((1, 16, 2))}
    environment = {
        "goal": torch.ones((1, 2)),
    }
    out = policy(states, actions, environment)
    print(out.size())
