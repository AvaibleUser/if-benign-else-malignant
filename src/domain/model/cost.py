import numpy as np

from domain.type.aliases import A, Y


def mse(output: A, expected: Y) -> np.floating:
    return np.mean((expected - output) ** 2)


def d_mse(output: A, expected: Y) -> np.ndarray:
    return output - expected


def binary_cross_entropy(output: A, expected: Y) -> np.floating:
    output = np.clip(output, 1e-15, 1 - 1e-15)
    return -np.mean(expected * np.log(output) + (1 - expected) * np.log(1 - output))


def d_binary_cross_entropy(output: A, expected: Y) -> np.ndarray:
    output = np.clip(output, 1e-15, 1 - 1e-15)
    return -(expected / output) + (1 - expected) / (1 - output)
