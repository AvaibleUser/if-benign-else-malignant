from attrs import define as attrs
from attrs import field


@attrs
class AlgorithmConfig:
    learning_rate: float = field(default=0.0005)
    epochs: int = field(default=200)
    train_size: float = field(default=0.65)
