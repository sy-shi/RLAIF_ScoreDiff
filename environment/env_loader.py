import environment.pacman.gym_wrapper
import environment.grid.gym_wrapper
import environment.grid.wrappers

from functools import partial
from ray.tune import register_env


# Use this function (with additional arguments if necessary) to additionally add wrappers to environments
def env_maker(config, env_name):
    env = None

    if env_name == "pacman":
        env = environment.pacman.gym_wrapper.GymPacman(config)
    elif env_name == "multi_grid":
        env = environment.grid.gym_wrapper.GymMultiGrid(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)

        # env = environment.grid.wrappers.GymCompatWrapper(env) # This wrapper is only needed if gym version is > 0.26!
        # env = environment.grid.wrappers.DoorUnlockBonus(env)

        if config.get("exploration_bonus", True):
            env = environment.grid.wrappers.ExplorationBonus(env)
            
        # env = environment.grid.wrappers.ActionBonus(env)
    elif env_name == "doubledoor":
        env = environment.grid.gym_wrapper.GymDoubleDoor(config)
        env = environment.grid.wrappers.FullyObsWrapper(env)
        env = environment.grid.wrappers.ActionMasking(env)

    else:
        raise("Unknown environment {}".format(env_name))

    return env


def register_envs():
    register_env("pacman", partial(env_maker, env_name = "pacman"))
    register_env("multi_grid", partial(env_maker, env_name = "multi_grid"))
    register_env("doubledoor", partial(env_maker, env_name = "doubledoor"))