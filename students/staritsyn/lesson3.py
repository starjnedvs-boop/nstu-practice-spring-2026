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
        self.x: np.ndarray | None = None
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_bias: np.ndarray = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.grad_weights = dy.T @ self.x
        self.grad_bias = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return self.grad_weights, self.grad_bias


class ReLULayer(Layer):
    def __init__(self):
        self.lm: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.lm = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.lm

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.sgm: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sgm = 1 / (1 + np.exp(-x))
        return self.sgm

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.sgm is None:
            raise ValueError("forward must be called before backward")
        return dy * self.sgm * (1 - self.sgm)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        self.y = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.y is None:
            raise ValueError("forward must be called before backward")
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
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [p for layer in self.layers for p in layer.parameters]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [g for layer in self.layers for g in layer.grad]


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        num_elements = self.x.size
        return 2 * (self.x - self.y) / num_elements


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y

        return -np.mean(self.y * np.log(self.x) + (1 - self.y) * np.log(1 - self.x))

    def backward(self) -> np.ndarray:
        return -(self.y / self.x - (1 - self.y) / (1 - self.x)) / self.x.shape[0]


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        self.grad = np.zeros_like(x)
        self.grad[np.arange(batch_size), y] = -1 / batch_size
        return -np.mean(x[np.arange(batch_size), y])

    def backward(self) -> np.ndarray:
        return self.grad


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_x = np.max(x, axis=-1, keepdims=True)
        log_softmax = x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))

        self.p = np.exp(log_softmax)
        self.y = y

        return -np.mean(log_softmax[np.arange(x.shape[0]), y])

    def backward(self) -> np.ndarray:
        batch_size = self.p.shape[0]
        grad = self.p.copy()

        grad[np.arange(batch_size), self.y] -= 1.0

        return grad / batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старицын Марк Вадимович, ПМ-35"

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
        np.split(x, idx, axis=0)

        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                pred = model.forward(x_batch)
                loss.forward(pred, y_batch)

                dy = loss.backward()
                model.backward(dy)

                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
