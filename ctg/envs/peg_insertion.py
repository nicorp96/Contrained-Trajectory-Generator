from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


def camera_config(uid, eye, target, width=128, height=128, fov=1.0, near=0.01, far=100):
    """CameraConfig looking from `eye` at `target` (world frame, metres; fov in rad)."""
    return CameraConfig(uid, sapien_utils.look_at(eye, target), width, height, fov, near, far)


@register_env("PegInsertionSide-Extended", max_episode_steps=100)
class PegInsertionSideConstrainedEnv(PegInsertionSideEnv):
    """PegInsertionSide registered under its own id; success comes from the base task.

    extra_cameras: list of camera_config() keyword dicts added to the default
    base camera, e.g. a side view of the hole.
    """

    def __init__(self, *args, extra_cameras=None, **kwargs):
        self.extra_cameras = [dict(c) for c in extra_cameras or []]
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [*super()._default_sensor_configs, *(camera_config(**c) for c in self.extra_cameras)]
