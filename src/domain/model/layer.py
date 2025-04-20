from typing import Iterable

import numpy as np

from domain.type.aliases import A, Activation, ErrorEntered, W, Z


class HiddenLayer:
    def __init__(self, layer_length: int, input_length: int, activation: Activation, d_activation: Activation, learning_rate: float):
        self.layer_length = layer_length
        self.weights = np.random.randn(
            layer_length, input_length) * np.sqrt(2 / input_length)
        self.biases = np.zeros((layer_length, 1))
        self.activation = activation
        self.d_activation = d_activation
        self.learning_rate = learning_rate

    def forward(self, input: np.ndarray) -> tuple[Z, A]:
        self.z = self.weights.dot(input) + self.biases
        self.a = self.activation(self.z)
        self.input = input
        return self.z, self.a

    def backward(self, error_next_layer: np.ndarray, weight_next_layer: np.ndarray, **_) -> tuple[ErrorEntered, W]:
        da_dz = self.d_activation(self.z)
        error_entered = weight_next_layer.T.dot(error_next_layer) * da_dz

        weights = self._learn(error_entered)

        return error_entered, weights

    def _learn(self, error_entered: ErrorEntered) -> W:
        dc_dw = error_entered.dot(self.input.T)
        dc_db = np.mean(error_entered)

        weights = self.weights

        self.biases -= self.learning_rate * dc_db
        self.weights -= self.learning_rate * dc_dw

        return weights


class OutputLayer(HiddenLayer):
    def backward(self, dc_da: np.ndarray, **_) -> tuple[ErrorEntered, W]:
        da_dz = self.d_activation(self.z)
        error_entered = dc_da * da_dz

        weights = self._learn(error_entered)

        return error_entered, weights


NeuralNetwork = Iterable[HiddenLayer | OutputLayer]
