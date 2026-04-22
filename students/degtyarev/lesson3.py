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

        scale = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-scale, scale, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-scale, scale, out_features).astype(np.float32)

        self._weight_gradient = np.zeros_like(self.weights)
        self._bias_gradient = np.zeros_like(self.bias)
        self._saved_input = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._saved_input = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self._weight_gradient = dy.T @ self._saved_input
        self._bias_gradient = np.sum(dy, axis=0)
        propagated_gradient = dy @ self.weights
        return propagated_gradient

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self._weight_gradient, self._bias_gradient)


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._positive_mask = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._positive_mask = (x > 0).astype(x.dtype)
        return x * self._positive_mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._positive_mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._saved_output = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._saved_output = 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))
        return self._saved_output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._saved_output * (1.0 - self._saved_output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self._probability_cache = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        max_per_row = np.max(x, axis=-1, keepdims=True)
        stabilized_logits = x - max_per_row
        exponentials = np.exp(stabilized_logits)
        exponential_sum = np.sum(exponentials, axis=-1, keepdims=True)

        self._probability_cache = exponentials / exponential_sum
        return stabilized_logits - np.log(exponential_sum)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        gradient_sum = np.sum(dy, axis=-1, keepdims=True)
        return dy - self._probability_cache * gradient_sum

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:
        self._layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        current_tensor = x
        for current_layer in self._layers:
            current_tensor = current_layer.forward(current_tensor)
        return current_tensor

    def backward(self, dy: np.ndarray) -> np.ndarray:
        current_gradient = dy
        for current_layer in reversed(self._layers):
            current_gradient = current_layer.backward(current_gradient)
        return current_gradient

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        collected_parameters = []
        for current_layer in self._layers:
            collected_parameters.extend(current_layer.parameters)
        return tuple(collected_parameters)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        collected_gradients = []
        for current_layer in self._layers:
            collected_gradients.extend(current_layer.grad)
        return tuple(collected_gradients)


class MSELoss(Loss):
    def __init__(self) -> None:
        self._prediction_cache = np.array([])
        self._target_cache = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._prediction_cache = x
        self._target_cache = y
        return np.array(np.mean((x - y) ** 2), dtype=np.float32)

    def backward(self) -> np.ndarray:
        return 2.0 * (self._prediction_cache - self._target_cache) / self._prediction_cache.size


class BCELoss(Loss):
    def __init__(self) -> None:
        self._prediction_cache = np.array([])
        self._target_cache = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._prediction_cache = x
        self._target_cache = y
        average_penalty = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
        return np.array(average_penalty)

    def backward(self) -> np.ndarray:
        row_count = self._prediction_cache.shape[0]
        return (
            (self._prediction_cache - self._target_cache)
            / (self._prediction_cache * (1 - self._prediction_cache))
            / row_count
        )


class NLLLoss(Loss):
    def __init__(self) -> None:
        self._stored_gradient = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        row_count = x.shape[0]
        class_indicator = np.zeros_like(x)
        class_indicator[np.arange(row_count), y] = 1
        self._stored_gradient = -class_indicator / row_count
        total_penalty = -np.sum(x * class_indicator) / row_count
        return np.array(total_penalty, dtype=np.float32)

    def backward(self) -> np.ndarray:
        return self._stored_gradient


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        self._stored_gradient = np.array([])
        self._stored_loss = np.array(0.0)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        row_count = x.shape[0]

        stabilized_logits = x - np.max(x, axis=-1, keepdims=True)
        log_probability_matrix = stabilized_logits - np.log(np.sum(np.exp(stabilized_logits), axis=-1, keepdims=True))

        class_indicator = np.zeros_like(x)
        class_indicator[np.arange(row_count), y] = 1

        self._stored_loss = -np.sum(log_probability_matrix * class_indicator) / row_count
        self._stored_gradient = (np.exp(log_probability_matrix) - class_indicator) / row_count
        return np.array(self._stored_loss)

    def backward(self) -> np.ndarray:
        return self._stored_gradient


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Дегтярев Кирилл Романович, ПМ-35"

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
        split_points = np.arange(batch_size, x.shape[0], batch_size)

        for _ in range(n_epoch):
            input_parts = np.split(x, split_points, axis=0)
            target_parts = np.split(y, split_points, axis=0)

            for feature_block, target_block in zip(input_parts, target_parts, strict=True):
                model_output = model.forward(feature_block)
                loss.forward(model_output, target_block)
                initial_gradient = loss.backward()
                model.backward(initial_gradient)

                for parameter_tensor, gradient_tensor in zip(model.parameters, model.grad, strict=True):
                    parameter_tensor += -lr * gradient_tensor
