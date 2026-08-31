# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState


def np_random(seed: Optional[int]) -> Tuple[RandomState, int]:
    """Set the seed for numpy's random generator (drop-in replacement for
    gym.utils.seeding.np_random, which returns a RandomState rather than the
    new-style Generator returned by gymnasium 1.x).

    Args:
        seed (Optional[int]):

    Returns:
        Tuple[RandomState, int]: Returns a tuple of random state and seed.
    """
    rng = RandomState(seed)
    if seed is None:
        seed = int(rng.randint(2 ** 31 - 1))
    assert isinstance(seed, int)
    return rng, seed
