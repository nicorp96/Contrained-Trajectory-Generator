from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env


@register_env("PegInsertionSide-Extended", max_episode_steps=100)
class PegInsertionSideConstrainedEnv(PegInsertionSideEnv):
    """PegInsertionSide registered under its own id; success comes from the base task."""
