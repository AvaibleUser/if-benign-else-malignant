from collections import namedtuple

import numpy as np
from sklearn.decomposition import PCA

from domain.model.activation import d_relu, d_sigmoid, relu, sigmoid
from domain.model.cost import (binary_cross_entropy, d_binary_cross_entropy,
                               d_mse, mse)
from domain.model.layer import HiddenLayer, NeuralNetwork, OutputLayer
from domain.type.aliases import A, ErrorEntered, UpdateChart, X, Y

Hidden = namedtuple("TopologyItem_Hidden", ("neurons",
                    "activation", "d_activation"), defaults=(None, relu, d_relu))
Output = namedtuple("TopologyItem_Output", ("neurons", "activation",
                    "d_activation"), defaults=(None, sigmoid, d_sigmoid))


def __create_neural_network(input_length: int, output_length: int, learning_rate: float) -> NeuralNetwork:
    topology = (Hidden(neurons=4), Hidden(neurons=10),
                Output(neurons=output_length))
    neural_network = []

    for layer in topology:
        Layer = HiddenLayer if isinstance(layer, Hidden) else OutputLayer
        neural_network.append(Layer(layer.neurons, input_length, layer.activation,
                                    layer.d_activation, learning_rate))

        input_length = layer.neurons

    return neural_network


def __predict(neural_network: NeuralNetwork, dataset: X) -> A:
    input: A = dataset
    for layer in neural_network:
        _, input = layer.forward(input)

    return input


def __learn(neural_network: NeuralNetwork, expected: Y, prediction: A) -> None:
    weights = None
    error: ErrorEntered = d_binary_cross_entropy(prediction, expected)
    layer: HiddenLayer | OutputLayer
    for layer in reversed(neural_network):
        error, weights = layer.backward(
            dc_da=error, error_next_layer=error, weight_next_layer=weights)


def __epoch(i: int, neural_network: NeuralNetwork, dataset: X, expected: Y, inverse_pca, callback: UpdateChart = None) -> None:
    prediction = __predict(neural_network, dataset)
    error = binary_cross_entropy(prediction, expected)

    __learn(neural_network, expected, prediction)

    accuracy = np.mean(np.equal(np.rint(prediction), expected))
    if callback is not None:
        rint_prediction = predict(neural_network, inverse_pca.T)
        callback(error, accuracy, rint_prediction)

    if i % 10 == 0:
        print(f"Epoch {i}, error: {error:.4f}, accuracy: {accuracy:.2%}")


def train(dataset: X, expected: Y, learning_rate: float, epochs: int, inverse_pca, callback: UpdateChart = None):
    neural_network = __create_neural_network(
        dataset.shape[0], 1, learning_rate)

    any(map(lambda i: __epoch(i, neural_network,
        dataset, expected, inverse_pca, callback), range(epochs)))

    return neural_network


def predict(neural_network: NeuralNetwork, dataset: X):
    prediction = __predict(neural_network, dataset)
    return np.rint(prediction)


def test(neural_network: NeuralNetwork, dataset: X, expected: Y):
    prediction = predict(neural_network, dataset)
    accuracy = np.mean(np.equal(prediction, expected))

    return accuracy
