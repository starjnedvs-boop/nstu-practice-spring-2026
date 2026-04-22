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

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_input = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dl_dw = dy.T @ self.x_input
        self.dl_db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self.dl_dw, self.dl_db]


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_out = np.maximum(x, 0)
        return self.x_out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * np.sign(self.x_out)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_out = 1 / (1 + np.exp(-x))
        return self.x_out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.x_out * (1 - self.x_out)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        self.x_out = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        return self.x_out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.x_out) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_inputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters)
        return parameters

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = (2 * (x - y)) / x.size
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        return self.grad


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = -((y / x) - (1 - y) / (1 - x)) / x.shape[0]
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self) -> np.ndarray:
        return self.grad


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = np.zeros_like(x)
        self.grad[np.arange(x.shape[0]), y] = -1 / x.shape[0]
        return -np.mean(x[np.arange(x.shape[0]), y])

    def backward(self) -> np.ndarray:
        return self.grad


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        logsoftmax = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        hot_y = np.zeros_like(x)
        hot_y[np.arange(x.shape[0]), y] = 1
        self.grad = (np.exp(logsoftmax) - hot_y) / x.shape[0]
        return -np.sum(logsoftmax * hot_y) / x.shape[0]

    def backward(self) -> np.ndarray:
        return self.grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузнецов Александр Павлович, ПМ-34"

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
        for _i in range(n_epoch):
            for _j in range(0, x.shape[0], batch_size):
                res = model.forward(x[_j : _j + batch_size])
                loss.forward(res, y[_j : _j + batch_size])
                model.backward(loss.backward())
                for params, grad in zip(model.parameters, model.grad, strict=True):
                    params -= grad * lr
