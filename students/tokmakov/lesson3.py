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


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()

        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

        self._x = np.empty(0, dtype=np.float32)
        self._weights_grad = np.zeros_like(self.weights)
        self._bias_grad = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = dy @ self.weights
        self._weights_grad = dy.T @ self._x
        self._bias_grad = np.sum(dy, axis=0)
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self._weights_grad, self._bias_grad]


class ReLULayer:
    def __init__(self) -> None:
        self._mask = np.empty(0, dtype=bool)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._y * (1 - self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis
        self._softmax = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self.axis, keepdims=True)
        x = x - x_max
        log_sum = np.log(np.sum(np.exp(x), axis=self.axis, keepdims=True))
        out = x - log_sum

        self._softmax = np.exp(out)
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dy_sum = np.sum(dy, axis=self.axis, keepdims=True)
        return dy - self._softmax * dy_sum

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, *layers: Layer) -> None:
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
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class MSELoss:
    def __init__(self) -> None:
        self._x = np.empty(0, dtype=np.float32)
        self._y = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        return 2 * (self._x - self._y) / self._x.size


class BCELoss:
    def __init__(self) -> None:
        self._x = np.empty(0, dtype=np.float32)
        self._y = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        return (self._x - self._y) / (self._x * (1 - self._x)) / batch_size


class NLLLoss:
    def __init__(self) -> None:
        self._x = np.empty(0, dtype=np.float32)
        self._y = np.empty(0, dtype=np.int64)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y

        batch_size = x.shape[0]
        mask = np.zeros_like(x)
        mask[np.arange(batch_size), y] = 1

        return -np.sum(x * mask) / batch_size

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        mask = np.zeros_like(self._x)
        mask[np.arange(batch_size), self._y] = 1

        return -mask / batch_size


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._x = np.empty(0, dtype=np.float32)
        self._y = np.empty(0, dtype=np.int64)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._y = y

        x_max = np.max(x, axis=-1, keepdims=True)
        x = x - x_max
        log_probs = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

        self._x = log_probs

        batch_size = x.shape[0]
        mask = np.zeros_like(x)
        mask[np.arange(batch_size), y] = 1

        return -np.sum(log_probs * mask) / batch_size

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        mask = np.zeros_like(self._x)
        mask[np.arange(batch_size), self._y] = 1

        return (np.exp(self._x) - mask) / batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Токмаков Дмитрий Евгеньевич, ПМ-31"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng=None) -> Layer:
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
        model: Layer,
        loss: Loss,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_epoch: int,
        batch_size: int,
    ) -> None:
        idx = np.arange(batch_size, x.shape[0], batch_size)

        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g
