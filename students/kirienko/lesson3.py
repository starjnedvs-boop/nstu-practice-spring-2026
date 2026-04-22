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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x = None
        self._dw = None
        self._db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self._dw = dy.T @ self._x
        self._db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self._dw is not None and self._db is not None
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        return dy * (self._x > 0)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dy * self._y * (1 - self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self) -> None:
        self._y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        x_shifted = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        self._y = x_shifted - log_sum_exp
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        softmax = np.exp(self._y)
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - softmax * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

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
        return [p for layer in self.layers for p in layer.parameters]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [g for layer in self.layers for g in layer.grad]


# Функции потерь (Loss functions)


class MSELoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        self._dy = 2 * (self._x - self._y) / self._x.size
        return self._dy


class BCELoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._dy: np.ndarray | None = None
        self._eps = 1e-15

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = np.clip(x, self._eps, 1 - self._eps)
        self._y = y
        return -np.mean(y * np.log(self._x) + (1 - y) * np.log(1 - self._x))

    def backward(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        batch_size = self._x.shape[0]
        self._dy = (self._x - self._y) / (self._x * (1 - self._x) * batch_size)
        return self._dy


class NLLLoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._hot_y: np.ndarray | None = None
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        batch_size = x.shape[0]
        self._hot_y = np.zeros_like(x)
        self._hot_y[np.arange(batch_size), y] = 1
        return -np.sum(x * self._hot_y) / batch_size

    def backward(self) -> np.ndarray:
        assert self._hot_y is not None and self._x is not None
        batch_size = self._x.shape[0]
        self._dy = -self._hot_y / batch_size
        return self._dy


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._hot_y: np.ndarray | None = None
        self._logprobs: np.ndarray | None = None
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        batch_size = x.shape[0]

        x_max = np.max(x, axis=-1, keepdims=True)
        x_shifted = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        self._logprobs = x_shifted - log_sum_exp

        self._hot_y = np.zeros_like(x)
        self._hot_y[np.arange(batch_size), y] = 1

        return -np.sum(self._logprobs * self._hot_y) / batch_size

    def backward(self) -> np.ndarray:
        assert self._logprobs is not None and self._hot_y is not None and self._x is not None
        batch_size = self._x.shape[0]
        softmax = np.exp(self._logprobs)
        self._dy = (softmax - self._hot_y) / batch_size
        return self._dy


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кириенко Илья Владимирович, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> Layer:
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
        n_samples = x.shape[0]

        for _ in range(n_epoch):
            # Разбиваем данные на батчи
            for i in range(0, n_samples, batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                # Обновление весов (градиентный спуск)
                for p, g in zip(model.parameters, model.grad, strict=True):
                    p -= lr * g
