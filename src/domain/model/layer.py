import numpy as np

from domain.model.activation import Activation

Z = np.ndarray
A = np.ndarray
W = np.ndarray
ErrorEntered = np.ndarray


class HiddenLayer:
    def __init__(self, layer_length: int, input_length: int, activation: Activation, d_activation: Activation, learning_rate: float):
        self.layer_length = layer_length
        self.weights = np.random.rand(input_length, layer_length)
        self.biases = np.random.rand(layer_length)
        self.activation = activation
        self.d_activation = d_activation
        self.learning_rate = learning_rate

    def forward(self, input: np.ndarray) -> tuple[Z, A]:
        self.z = self.weights.dot(input) + self.biases
        self.a = self.activation(self.z)
        self.input = input
        return self.z, self.a

    def backward(self, error_next_layer: np.ndarray, weight_next_layer: np.ndarray) -> tuple[ErrorEntered, W]:
        da_dz = self.d_activation(self.z)
        error_entered = weight_next_layer.T.dot(error_next_layer) * da_dz

        weights = self.__learn(error_entered)

        return error_entered, weights

    def __learn(self, error_entered: ErrorEntered) -> W:
        dc_dw = self.input.T.dot(error_entered)
        dc_db = np.mean(error_entered, axis=0, keepdims=True)

        weights = self.weights

        self.biases -= self.learning_rate * dc_db
        self.weights -= self.learning_rate * dc_dw

        return weights


class OutputLayer(HiddenLayer):
    def backward(self, dc_da: np.ndarray) -> tuple[ErrorEntered, W]:
        da_dz = self.d_activation(self.z)
        error_entered = dc_da * da_dz

        weights = self.__learn(error_entered)

        return error_entered, weights


Layer = HiddenLayer | OutputLayer
