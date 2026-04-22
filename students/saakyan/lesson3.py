from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backward(self, dy: np.ndarray) -> np.ndarray: ...

    @property
    def parameters(self) -> Sequence[np.ndarray]: ...

    @property
    def grad(self) -> Sequence[np.ndarray]: ...


class Loss(Protocol):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def backward(self) -> np.ndarray: ...


class LinearLayer(Layer):
    weights: np.ndarray
    bias: np.ndarray
    x: np.ndarray
    dw: np.ndarray
    db: np.ndarray

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()

        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dw = dy.T @ self.x
        self.db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.dw, self.db)


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = np.maximum(x, 0)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * np.sign(self.y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.y * (1 - self.y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        self.y = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.y) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters)
        return tuple(parameters)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grad)
        return tuple(gradients)


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.dy = 2 * (x - y) / x.size
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        return self.dy


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-12, 1 - 1e-12)
        batch_size = x.shape[0]
        self.dy = (x - y) / (x * (1 - x)) / batch_size
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self) -> np.ndarray:
        return self.dy


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        self.dy = np.zeros_like(x)
        self.dy[np.arange(batch_size), y] = -1 / batch_size
        return -np.sum(x[np.arange(batch_size), y]) / batch_size

    def backward(self) -> np.ndarray:
        return self.dy


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        logprobs = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))

        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1

        self.dy = (np.exp(logprobs) - hot_y) / batch_size
        return -np.sum(logprobs * hot_y) / batch_size

    def backward(self) -> np.ndarray:
        return self.dy


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Саакян Айк Алексанович, ПМ-34"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> Layer:
        return LinearLayer(in_features, out_features, rng)

    @staticmethod
    def create_relu_layer() -> Layer:
        return ReLULayer()

    @staticmethod
    def create_sigmoid_layer() -> Layer:
        return SigmoidLayer()

    @staticmethod
    def create_logsoftmax_layer() -> Layer:
        return LogSoftmaxLayer()

    @staticmethod
    def create_model(*layers: Layer) -> Layer:
        return Model(*layers)

    @staticmethod
    def create_mse_loss() -> Loss:
        return MSELoss()

    @staticmethod
    def create_bce_loss() -> Loss:
        return BCELoss()

    @staticmethod
    def create_nll_loss() -> Loss:
        return NLLLoss()

    @staticmethod
    def create_cross_entropy_loss() -> Loss:
        return CrossEntropyLoss()

    @staticmethod
    def train_model(
        model: Layer, loss: Loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int
    ) -> None:
        if batch_size <= 0 or batch_size > x.shape[0]:
            batch_size = x.shape[0]

        for _ in range(n_epoch):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                for parameter, grad in zip(model.parameters, model.grad, strict=True):
                    parameter -= lr * grad
