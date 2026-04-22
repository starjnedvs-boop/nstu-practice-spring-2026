from __future__ import annotations

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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        k = float(np.sqrt(1.0 / in_features))
        self.w = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.b = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x: np.ndarray | None = None
        self._dw: np.ndarray | None = None
        self._db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.w.T + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        x = self._x
        self._dw = dy.T @ x
        self._db = np.sum(dy, axis=0)
        return dy @ self.w

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.w, self.b)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        if self._dw is None or self._db is None:
            return ()
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.maximum(x, 0)
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        return dy * np.sign(self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = _sigmoid(x)
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        y = self._y
        return dy * y * (1 - y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = _log_softmax(x)
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        y = self._y
        return dy - (np.exp(y) * np.sum(dy, axis=-1, keepdims=True))

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, layers: tuple[Layer, ...]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        grad = dy
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params: list[np.ndarray] = []
        for layer in self.layers:
            params.extend(list(layer.parameters))
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads: list[np.ndarray] = []
        for layer in self.layers:
            grads.extend(list(layer.grad))
        return tuple(grads)


class MSELoss:
    def __init__(self) -> None:
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        loss = np.mean((x - y) ** 2)
        self._dy = 2 * (x - y) / x.size
        return loss

    def backward(self) -> np.ndarray:
        if self._dy is None:
            raise RuntimeError("forward must be called before backward")
        return self._dy


class BCELoss:
    def __init__(self) -> None:
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        loss = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
        self._dy = (x - y) / (x * (1 - x)) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        if self._dy is None:
            raise RuntimeError("forward must be called before backward")
        return self._dy


class NLLLoss:
    def __init__(self) -> None:
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1
        loss = -np.sum(x * hot_y) / batch_size
        self._dy = -hot_y / batch_size
        return loss

    def backward(self) -> np.ndarray:
        if self._dy is None:
            raise RuntimeError("forward must be called before backward")
        return self._dy


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._dy: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1
        logprobs = _log_softmax(x)
        loss = -np.sum(logprobs * hot_y) / batch_size
        self._dy = (np.exp(logprobs) - hot_y) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        if self._dy is None:
            raise RuntimeError("forward must be called before backward")
        return self._dy


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Мелиди Мирон Евстафьевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> Layer:
        return LinearLayer(in_features, out_features, rng or np.random.default_rng())

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
        return Model(tuple(layers))

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
