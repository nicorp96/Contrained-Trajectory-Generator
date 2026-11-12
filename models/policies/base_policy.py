import os
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from common.get_class import get_class_dict
from common.utils import load_config
from models.utils.guidance_robot import BaseGuidance
from models.utils.normalizer import DictNormalizer
from models.utils.robot_constraints import RobotConstraints


class Policy:
    """
    Loads a trained diffusion trajectory model from a checkpoint.

    kwargs:
      guidance:     dict overriding config["model"]["guidance"] (e.g. constraint_eta,
                    project_final, guidance_scale) for inference-time ablations
      constraints:  dict overriding config["constraints"]
    """

    def __init__(self, ckpt_path, device, only_actions=True, **kwargs):
        self.device = torch.device(device)
        self.only_actions = only_actions
        self.model = None
        self.ema_model = None
        self.checkpoint = None
        self.config = None
        self.noise_scheduler = None
        self.normalizer_state, self.normalizer_goal, self.normalizer_obs = None, None, None
        self.__load_ckp__(ckpt_path)
        self.setup_model()
        self.setup_scheduler()
        self.setup_normalizer()
        self.sequence_len = self.config["dataset"]["sequence_len"]
        self.goal_indices = self.config.get("goal_indices", None)
        self.start_indices = self.config.get("start_indices", None)
        self.inpainting = self.config.get("inpainting", True)

        guidance_cfg = {**self.config["model"]["guidance"], **kwargs.get("guidance", {})}
        self.guidance_algo: BaseGuidance = get_class_dict(guidance_cfg)
        constraints_cfg = kwargs.get("constraints", self.config.get("constraints"))
        self.constraints = (
            RobotConstraints(constraints_cfg, self.config["dataset"]["state_shapes"])
            if constraints_cfg is not None
            else None
        )
        self.num_inference_steps = self.config["model"]["num_inference_steps"]

    def __load_ckp__(self, ckpt_path):
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        # the config is stored in the checkpoint; files/config.yaml is the fallback
        config_f = checkpoint_path.parent.parent / "files" / "config.yaml"
        self.config = self.checkpoint.get("config") or load_config(config_f)

    def setup_model(self):
        sd = self.checkpoint["state_dicts"]
        self.model = get_class_dict(self.config["model"])
        self.model.load_state_dict(sd["model_state"])
        self.model = self.model.to(self.device).eval()
        if sd.get("ema_state") is not None:
            self.ema_model = get_class_dict(self.config["model"])
            self.ema_model.load_state_dict(sd["ema_state"])
            self.ema_model = self.ema_model.to(self.device).eval()

    def setup_normalizer(self):
        shapes = self.config["dataset"]
        self.normalizer_state = DictNormalizer(shapes["state_shapes"])
        self.normalizer_goal = DictNormalizer(shapes["goal_shapes"])
        self.normalizer_obs = DictNormalizer(shapes.get("obs_shapes") or {})
        saved = self.checkpoint["state_dicts"].get("normalizer")
        if saved is not None:
            self.normalizer_state.load_stats(saved["normalizer_state"])
            self.normalizer_goal.load_stats(saved["normalizer_goal"])
            self.normalizer_obs.load_stats(saved["normalizer_obs"])
        else:
            # older checkpoints: statistics from the dataset's stats file
            from dataset.utils import get_ds_from_cfg

            train_dataset, _ = get_ds_from_cfg(shapes)
            base = getattr(train_dataset, "dataset", train_dataset)
            self.normalizer_state, self.normalizer_goal, self.normalizer_obs = (
                base.get_normalizers_from_file()
            )
        for normalizer in (self.normalizer_state, self.normalizer_goal, self.normalizer_obs):
            normalizer.to(self.device)
            normalizer.set_device(self.device)

    def setup_scheduler(self):
        scheduler_cfg = self.config["model"]["noise_scheduler"]
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

    def reset(self):
        """Call at the start of every episode."""
        pass

    def conditional_sample(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, obs):
        raise NotImplementedError
