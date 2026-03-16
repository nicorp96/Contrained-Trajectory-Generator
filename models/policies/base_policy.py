from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
import torch

from common.get_class import get_class_dict
from dataset.utils import get_ds_from_cfg
from models.utils.guidance_robot import BaseGuidance


class Policy:
    def __init__(self, ckpt_path, device, only_actions=True, **kwargs):
        self.device = device
        self.only_actions = only_actions
        self.config_policy = None
        self.model = None
        self.ema_model = None
        self.checkpoint = None
        self.config = None
        self.diff_model = None
        self.noise_scheduler = None
        self.normalizer_state, self.normalizer_goal, self.normalizer_obs = (
            None,
            None,
            None,
        )
        self.__load_ckp__(ckpt_path)
        self.setup_model()
        self.setup_scheduler()
        self.setup_normalizer()
        self.sequence_len = self.config["dataset"]["sequence_len"]
        self.goal_indices = self.config.get("goal_indices", None)
        self.start_indices = self.config.get("start_indices", None)
        self.inpainting = self.config.get("inpainting", True)
        self.guidance_algo: BaseGuidance = get_class_dict(
            self.config["model"]["guidance"]
        )
        self.num_inference_steps = self.config["model"]["num_inference_steps"]

    def __load_ckp__(self, ckpt_path):
        checkpoint_path = ckpt_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint["config"]
        # self.config_policy = self.config["policy"]

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        self.model.load_state_dict(self.checkpoint["state_dicts"]["model_state"])
        self.model = self.model.to(self.device)
        if self.checkpoint["state_dicts"]["ema_state"] is not None:
            self.ema_model = get_class_dict(self.config["model"])
            self.ema_model.load_state_dict(self.checkpoint["state_dicts"]["ema_state"])
            self.ema_model = self.ema_model.to(self.device)

    # TODO get normalizer from save ckp states
    def setup_normalizer(self):
        dataset_dict = self.config["dataset"]
        train_dataset, validation_dataset = get_ds_from_cfg(dataset_dict)
        self.train_dataset = train_dataset
        # TODO: load normalizer from dict
        self.normalizer_state, self.normalizer_goal, self.normalizer_obs = (
            self.train_dataset.dataset.get_normalizers_from_file()
        )

    def setup_scheduler(self):
        scheduler_cfg = self.config["model"][
            "noise_scheduler"
        ]  # TODO: use target from config
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_cfg["num_train_timesteps"],
            beta_start=scheduler_cfg["beta_start"],
            beta_end=scheduler_cfg["beta_end"],
            beta_schedule=scheduler_cfg["beta_schedule"],
            clip_sample=scheduler_cfg["clip_sample"],
            set_alpha_to_one=scheduler_cfg["set_alpha_to_one"],
            steps_offset=scheduler_cfg["steps_offset"],
            prediction_type=scheduler_cfg["prediction_type"],
        )

    def conditional_sample(self):
        raise NotImplementedError

    def __call__(self, obs):
        raise NotImplementedError
