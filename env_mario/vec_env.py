from typing import Any
import numpy as np
from gym.vector import AsyncVectorEnv

from .env import MarioEnvironment

def make_env(seed: int, **env_kwargs: Any):
    """
    Returns a function that creates a MarioEnvironment instance with a specific seed and additional arguments.

    Args:
        seed (int): Seed for environment reproducibility.
        **env_kwargs: Additional keyword arguments for MarioEnvironment.

    Returns:
        Callable[[], MarioEnvironment]: A function that creates a new MarioEnvironment instance.
    """
    def thunk():
        return MarioEnvironment(seed=seed, **env_kwargs)
    return thunk


def build_vec_env(num_envs: int, base_seed: int | None = None, **env_kwargs: Any):
    """
    Creates an asynchronous vectorized environment with independent seeds.

    Args:
        num_envs (int): Number of parallel environments to create.
        base_seed (int | None, optional): Base seed for reproducibility. If None, a random seed is used.
        **env_kwargs: Additional keyword arguments passed to each MarioEnvironment.

    Returns:
        AsyncVectorEnv: An asynchronous vectorized environment containing the specified number of MarioEnvironment instances.
    """
    ss = np.random.SeedSequence(base_seed)
    child_ss = ss.spawn(num_envs)
    seeds = [int(np.random.default_rng(s).integers(0, 2**31 - 1)) for s in child_ss]

    thunks = [make_env(seed=s, **env_kwargs) for s in seeds]
    return AsyncVectorEnv(thunks)