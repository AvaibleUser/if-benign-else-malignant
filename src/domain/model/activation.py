import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def d_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))
