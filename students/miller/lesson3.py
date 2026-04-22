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
    def __init__(self) -> None: ...

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def backward(self) -> np.ndarray: ...


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights.T) + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dweights = np.dot(dy.T, self.input)
        self.dbias = np.sum(dy, axis=0)
        return np.dot(dy, self.weights)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self.dweights, self.dbias]


class ReLULayer(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError

        return dy * (self.input > 0)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError

        return dy * self.output * (1 - self.output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.output = None
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x

        axis = -1
        max_vals = np.max(x, axis=axis, keepdims=True)
        shifted = x - max_vals

        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
        self.output = shifted - np.log(sum_exp)

        return self.output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError

        axis = -1
        softmax = np.exp(self.output)
        grad = dy - softmax * np.sum(dy, axis=axis, keepdims=True)
        return grad

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
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params = ()
        for layer in self.layers:
            params = params + tuple(layer.parameters)
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = ()
        for layer in self.layers:
            grads = grads + tuple(layer.grad)
        return grads


class MSELoss(Loss):
    def __init__(self) -> None:
        self.x: np.ndarray = np.empty(0, dtype=np.float32)
        self.y: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.array(np.mean((x - y) ** 2))

    def backward(self) -> np.ndarray:
        n = self.x.size
        return 2 * (self.x - self.y) / n


class BCELoss(Loss):
    def __init__(self, eps: float = 1e-8) -> None:
        self.x: np.ndarray = np.empty(0, dtype=np.float32)
        self.y: np.ndarray = np.empty(0, dtype=np.float32)
        self.eps = eps

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        x = np.clip(x, self.eps, 1 - self.eps)
        return np.array(-np.mean(y * np.log(x) + (1 - y) * np.log(1 - x)))

    def backward(self) -> np.ndarray:
        x = np.clip(self.x, self.eps, 1 - self.eps)
        n = self.x.size
        return (x - self.y) / (x * (1 - x)) / n


class NLLLoss(Loss):
    def __init__(self) -> None:
        self.x: np.ndarray = np.empty(0, dtype=np.float32)
        self.y: np.ndarray = np.empty(0, dtype=np.int64)
        self.grad: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y

        return np.array(-np.mean(x[np.arange(x.shape[0]), y]), dtype=np.float32)

    def backward(self) -> np.ndarray:
        N = self.x.shape[0]

        grad = np.zeros_like(self.x, dtype=np.float32)
        grad[np.arange(N), self.y] = -1.0 / N

        return grad


class CELoss(Loss):
    def __init__(self, eps: float = 1e-8) -> None:
        self.x: np.ndarray = np.empty(0, dtype=np.float32)
        self.y: np.ndarray = np.empty(0, dtype=np.int64)
        self.eps = eps

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y

        x = np.clip(x, self.eps, 1 - self.eps)
        return np.array(-np.mean(np.log(x[np.arange(x.shape[0]), y])))

    def backward(self) -> np.ndarray:
        n = self.x.shape[0]

        grad = np.zeros_like(self.x)
        grad[np.arange(n), self.y] = -1.0 / (self.x[np.arange(n), self.y] * n)

        return grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Миллер Игорь Владиславович, ПМ-31"

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
        return CELoss()
