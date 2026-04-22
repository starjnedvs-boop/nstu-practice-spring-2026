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

        self._input_cache: np.ndarray | None = None
        self._grad_weights: np.ndarray | None = None
        self._grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:

        self._grad_weights = dy.T @ self._input_cache
        self._grad_bias = np.sum(dy, axis=0)

        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self._grad_weights is not None
        assert self._grad_bias is not None
        return [self._grad_weights, self._grad_bias]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.maximum(0, x)
        self._input_cache = x
        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._input_cache is not None
        mask = self._input_cache > 0
        dx = dy * mask
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = 1 / (1 + np.exp(-x))
        self._output_cache = output
        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._output_cache is not None
        sigmoid_x = self._output_cache
        dx = dy * sigmoid_x * (1 - sigmoid_x)
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self._softmax_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        softmax = exp_x / sum_exp
        log_softmax = x - x_max - np.log(sum_exp)
        self._softmax_cache = softmax
        return log_softmax

    def backward(self, dy: np.ndarray) -> np.ndarray:
        softmax = self._softmax_cache
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        dx = dy - softmax * sum_dy
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
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
        size = x.size

        diff = x - y
        sq = diff**2

        self.grad = 2 * diff / size

        return np.mean(sq)

    def backward(self) -> np.ndarray:
        return self.grad


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        sum = y * np.log(x) + (1 - y) * np.log(1 - x)

        self.grad = (x - y) / (x * (1 - x) * batch_size)

        return -np.mean(sum)

    def backward(self) -> np.ndarray:
        return self.grad


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
        batch_size = x.shape[0]

        max_x = np.max(x, axis=1, keepdims=True)

        log_sum_exp = np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))

        log_probs = x - max_x - log_sum_exp

        softmax = np.exp(log_probs)

        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1

        self.grad = (softmax - hot_y) / batch_size

        return -np.sum(log_probs * hot_y) / batch_size

    def backward(self) -> np.ndarray:
        return self.grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Большанин Егор Андреевич, ПМ-33"

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

        n = x.shape[0]

        for _ in range(n_epoch):
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)

                x_batch = x[i:end]
                y_batch = y[i:end]

                pred = model.forward(x_batch)

                loss.forward(pred, y_batch)

                grad_pred = loss.backward()

                model.backward(grad_pred)

                for param, grad in zip(model.parameters, model.grad, strict=False):
                    param -= lr * grad
