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
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

        self.dw: np.ndarray
        self.db: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self.x

        self.dw = dy.T @ x
        self.db = np.sum(dy, axis=0)
        dx = dy @ self.weights

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.dw, self.db)


class ReLULayer(Layer):
    def __init__(self):
        self.x: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        y = np.maximum(self.x, 0)
        return dy * np.sign(y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.y: np.ndarray

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
    def __init__(self):
        self.y: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        self.y = x_shift - np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - (np.exp(self.y) * np.sum(dy, axis=-1, keepdims=True))

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return tuple(grads)


class MSELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.array(np.mean((x - y) ** 2))

    def backward(self) -> np.ndarray:
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        eps = 1e-7
        x_clipped = np.clip(x, eps, 1 - eps)
        return np.array(-np.mean(y * np.log(x_clipped) + (1 - y) * np.log(1 - x_clipped)))

    def backward(self) -> np.ndarray:
        eps = 1e-7
        x_clipped = np.clip(self.x, eps, 1 - eps)
        return (x_clipped - self.y) / (x_clipped * (1 - x_clipped)) / self.x.shape[0]


class NLLLoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        batch_size = x.shape[0]
        loss = -np.sum(x[np.arange(batch_size), y]) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        batch_size = self.x.shape[0]
        dy = np.zeros_like(self.x)
        dy[np.arange(batch_size), self.y] = -1.0 / batch_size
        return dy


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray
        self.logprobs: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        self.logprobs = x_shift - np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))

        batch_size = x.shape[0]
        loss = -np.sum(self.logprobs[np.arange(batch_size), y]) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        batch_size = self.x.shape[0]
        hot_y = np.zeros_like(self.x)
        hot_y[np.arange(batch_size), self.y] = 1
        return (np.exp(self.logprobs) - hot_y) / batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Гросс Кирилл Дмитриевич, ПМ-33"

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
        idx = np.arange(batch_size, x.shape[0], batch_size)

        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g
