from typing import Callable

from numpy import ndarray

X = ndarray
Y = ndarray
Z = ndarray
A = ndarray
W = ndarray

ErrorEntered = ndarray

Cost = Callable[[ndarray, ndarray], ndarray]

Activation = Callable[[ndarray], ndarray]
