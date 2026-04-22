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

        self.cache: np.ndarray
        self.dw: np.ndarray
        self.db: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self.cache

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
        self.cache: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * (self.cache > 0).astype(np.float32)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.cache: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = 1 / (1 + np.exp(-x))
        return self.cache

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.cache * (1 - self.cache)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.cache: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        self.cache = x_shift - np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
        return self.cache

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - (np.exp(self.cache) * np.sum(dy, axis=-1, keepdims=True))

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
        self.pred: np.ndarray | None = None
        self.target: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.pred = x
        self.target = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        # Добавляем проверку, чтобы линтер не ругался
        if self.pred is None or self.target is None:
            raise RuntimeError("forward must be called before backward")
        return 2 * (self.pred - self.target) / self.pred.size


class BCELoss(Loss):
    def __init__(self):
        self.pred: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.eps = 1e-13

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.pred = np.clip(x, self.eps, 1 - self.eps)
        self.target = y
        return -np.mean(y * np.log(self.pred) + (1 - y) * np.log(1 - self.pred))

    def backward(self) -> np.ndarray:
        if self.pred is None or self.target is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.pred.shape[0]
        return (self.pred - self.target) / (self.pred * (1 - self.pred)) / batch_size


class NLLLoss(Loss):
    def __init__(self):
        self.logits: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.logits = x
        self.labels = y.astype(np.int32)
        batch_size = x.shape[0]
        return -np.mean(x[np.arange(batch_size), self.labels])

    def backward(self) -> np.ndarray:
        if self.logits is None or self.labels is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.logits.shape[0]
        grad = np.zeros_like(self.logits)
        grad[np.arange(batch_size), self.labels] = -1 / batch_size
        return grad


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.logsoftmax: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        self.logsoftmax = x_shift - np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
        self.labels = y.astype(np.int32)
        batch_size = x.shape[0]
        return -np.mean(self.logsoftmax[np.arange(batch_size), self.labels])

    def backward(self) -> np.ndarray:
        if self.logsoftmax is None or self.labels is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.logsoftmax.shape[0]
        grad = np.exp(self.logsoftmax).copy()
        grad[np.arange(batch_size), self.labels] -= 1
        return grad / batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Наумов Дмитрий Сергеевич, ПМ-33"

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
