
from typing import Callable

import numpy as np

Cost = Callable[[np.ndarray, np.ndarray], np.ndarray]


def binary_cross_entropy(output: np.ndarray, expected: np.ndarray) -> np.floating:
    return -np.mean(expected * np.log(output) + (1 - expected) * np.log(1 - output))


def d_binary_cross_entropy(output: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return -(expected / output) + (1 - expected) / (1 - output)
