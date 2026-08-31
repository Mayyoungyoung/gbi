# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Collection of types used in the env."""


from typing import Dict, Union

import numpy as np
from gymnasium.vector.async_vector_env import AsyncVectorEnv  # gym shim

from mtrl.utils.types import TensorType

EnvType = AsyncVectorEnv
TaskObsType = Union[TensorType]
ActionType = Union[str, int, float, np.ndarray]
EnvObsType = Union[TensorType]
ObsType = Dict[str, Union[EnvObsType, TaskObsType]]
