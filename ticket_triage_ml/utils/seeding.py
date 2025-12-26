"""Random seed utilities for reproducibility."""

import os
import random

import numpy as np
from loguru import logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (if available).

    Args:
        seed: Random seed value.
    """
    logger.debug(f"Setting random seed: {seed}")

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except ImportError:
        pass


def get_random_seed() -> int:
    """Generate a random seed.

    Returns:
        Random integer seed.
    """
    return random.randint(0, 2**32 - 1)
