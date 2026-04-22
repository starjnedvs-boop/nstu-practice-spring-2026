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
        self.input = x
        return self.bias + x @ self.weights.T

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.grad_weights = dy.T @ self.input
        self.grad_bias = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return self.grad_weights, self.grad_bias


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(x, 0)
        return self.output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * np.sign(self.output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.output * (1 - self.output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        max_x = np.max(x, axis=-1, keepdims=True)
        self.output = x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
        return self.output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.output) * np.sum(dy, axis=-1, keepdims=True)

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
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.mean(np.square(x - y))

    def backward(self) -> np.ndarray:
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

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
        self.y = y
        batch_size = x.shape[0]
        max_x = np.max(x, -1, keepdims=True)
        logsoftmax = x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
        self.p = np.exp(logsoftmax)
        return -np.mean(logsoftmax[np.arange(batch_size), y])

    def backward(self) -> np.ndarray:
        batch_size = self.p.shape[0]
        one_hot = np.zeros_like(self.p)
        one_hot[np.arange(batch_size), self.y] = 1
        return (self.p - one_hot) / batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Пантеева Валентина Ивановна, ПМ-33"

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
        n_samples = x.shape[0]

        for _ in range(n_epoch):
            for start in range(0, n_samples, batch_size):
                x_batch = x[start : start + batch_size]
                y_batch = y[start : start + batch_size]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)
                dloss = loss.backward()

                model.backward(dloss)

                params = model.parameters
                grads = model.grad

                for param, grad in zip(params, grads, strict=True):
                    param -= lr * grad
