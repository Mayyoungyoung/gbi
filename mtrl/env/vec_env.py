# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict

import numpy as np
import torch
# gym shim: CTPG was written against gym 0.18-0.21 vector API. The MetaWorld
# envs installed on this machine are gymnasium-based, so we use the gymnasium
# AsyncVectorEnv here and re-pack its 5-tuple step protocol into the old
# 4-tuple gym protocol expected by the rest of the codebase.
from gymnasium.vector.async_vector_env import AsyncVectorEnv


class VecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def reset(self):
        multitask_obs, infos = super().reset()
        return _cast_multitask_obs(multitask_obs)

    def step(self, actions):
        multitask_obs, reward, terminated, truncated, infos = super().step(actions)
        multitask_obs, reward, done, info_list = _repack_gymnasium_step(
            self, multitask_obs, reward, terminated, truncated, infos
        )
        return _cast_multitask_obs(multitask_obs), reward, done, info_list

    def seed(self, seed=None):
        return _seed_async_env(self, seed)


def _cast_multitask_obs(multitask_obs):
    return {key: torch.tensor(value) for key, value in multitask_obs.items()}


def _seed_async_env(vec_env, seed):
    """Seed each worker env with seed + i (mirrors gym 0.21 AsyncVectorEnv.seed).
    gymnasium 1.x removed VectorEnv.seed, so we send `_call` messages directly."""
    if seed is None:
        seeds = [None] * vec_env.num_envs
    else:
        seeds = [seed + i for i in range(vec_env.num_envs)]
    for pipe, s in zip(vec_env.parent_pipes, seeds):
        pipe.send(("_call", ("seed", (s,), {})))
    results = []
    for pipe in vec_env.parent_pipes:
        result, success = pipe.recv()
        if not success:
            error = vec_env.error_queue.get(timeout=60)
            raise error
        results.append(result)
    return results


def _repack_gymnasium_step(vec_env, multitask_obs, reward, terminated, truncated, infos):
    """Convert gymnasium 5-tuple step into the old gym 4-tuple step:
    (obs, reward, done, info_list)."""
    done = np.logical_or(terminated, truncated)
    info_list = []
    for i in range(vec_env.num_envs):
        info = infos[i] if i in infos else {}
        # gymnasium autoreset steps carry a bare reset info dict; fill the keys
        # that the CTPG training/eval loops read unconditionally.
        info.setdefault("success", 0.0)
        info.setdefault("env_step", 0)
        info_list.append(info)
    return multitask_obs, reward, done, info_list


class MetaWorldVecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        self.task_obs = torch.arange(self.num_envs)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def reset(self):
        env_obs, infos = super().reset()
        return self.create_multitask_obs(env_obs=env_obs)

    def step(self, actions):
        env_obs, reward, terminated, truncated, infos = super().step(actions)
        env_obs, _, done, info_list = _repack_gymnasium_step(
            self, env_obs, reward, terminated, truncated, infos
        )
        return self.create_multitask_obs(env_obs=env_obs), reward, done, info_list

    def seed(self, seed=None):
        return _seed_async_env(self, seed)

    def create_multitask_obs(self, env_obs):
        return {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs}


class GymExtensionsVecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        self.task_obs = torch.arange(self.num_envs)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def reset(self):
        env_obs, infos = super().reset()
        return self.create_multitask_obs(env_obs=env_obs)

    def step(self, actions):
        env_obs, reward, terminated, truncated, infos = super().step(actions)
        env_obs, _, done, info_list = _repack_gymnasium_step(
            self, env_obs, reward, terminated, truncated, infos
        )
        return self.create_multitask_obs(env_obs=env_obs), reward, done, info_list

    def seed(self, seed=None):
        return _seed_async_env(self, seed)

    def create_multitask_obs(self, env_obs):
        return {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs}