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
        self.x = None
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dw = dy.T @ self.x
        self.db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return self.dw, self.db


class ReLULayer(Layer):
    def __init__(self) -> None:
        self.mask: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self.sgm_out: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sgm_out = 1 / (1 + np.exp(-x))
        return self.sgm_out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.sgm_out * (1 - self.sgm_out)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self.softmax: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_solu = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_solu)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        self.softmax = exp_x / sum_exp
        return x_solu - np.log(sum_exp)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - self.softmax * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layer: Layer):
        self.layers = layer

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


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = 2.0 * (x - y) / x.size
        return np.mean(np.square(x - y))

    def backward(self) -> np.ndarray:
        return self.grad


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = -(y / x - (1.0 - y) / (1.0 - x)) / x.shape[0]
        return -np.mean(y * np.log(x) + (1.0 - y) * np.log(1.0 - x))

    def backward(self) -> np.ndarray:
        return self.grad


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = np.zeros_like(x)
        self.grad[np.arange(x.shape[0]), y] = -1.0 / x.shape[0]
        return -np.mean(x[np.arange(x.shape[0]), y])

    def backward(self) -> np.ndarray:
        return self.grad


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.grad = np.zeros_like(x)
        self.grad[np.arange(x.shape[0]), y] = 1.0

        x_solu = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_solu)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        log_sum = np.log(sum_exp)
        log_probabl = x_solu - log_sum
        softmax = np.exp(log_probabl)

        self.grad = (softmax - self.grad) / x.shape[0]

        return -np.mean(log_probabl[np.arange(x.shape[0]), y])

    def backward(self) -> np.ndarray:
        return self.grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Урывский Александр Александрович, ПМ-31"

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
        for _ in range(n_epoch):
            for i in range(0, x.shape[0], batch_size):
                sol = model.forward(x[i : i + batch_size])
                loss.forward(sol, y[i : i + batch_size])
                model.backward(loss.backward())
                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= grad * lr
