import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        mse = np.mean((pred - y) ** 2)
        return float(mse)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(1 - (np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        n = len(x)
        dw = -2 / n * (x.T @ (y - pred))
        db = -2 / n * np.sum(y - pred)
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features).astype(np.float64)
        self.bias = np.array(0.0, dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean(-y * np.log(pred) - (1 - y) * np.log(1 - pred)))

    def metric(self, x: np.ndarray, y: np.ndarray, mtype: str = "precision") -> float:
        pred = self.predict(x)

        pred_bin = (pred >= 0.5).astype(int)

        tp = np.sum((pred_bin == 1) & (y == 1))
        fp = np.sum((pred_bin == 1) & (y == 0))
        fn = np.sum((pred_bin == 0) & (y == 1))
        tn = np.sum((pred_bin == 0) & (y == 0))
        if mtype == "accuracy":
            return (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0.0
        if mtype == "precision":
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if mtype == "recall":
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if mtype == "F1":
            return tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0.0
        if mtype == "AUROC":
            pos_scores = pred[y == 1]
            neg_scores = pred[y == 0]

            n_pos = len(pos_scores)
            n_neg = len(neg_scores)

            if n_pos == 0 or n_neg == 0:
                return 0.5

            n_correct = 0
            total_pairs = n_pos * n_neg

            for pos_score in pos_scores:
                for neg_score in neg_scores:
                    if pos_score > neg_score:
                        n_correct += 1
                    elif pos_score == neg_score:
                        n_correct += 0.5

            return n_correct / total_pairs
        return 0

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        n = np.float64(len(x))
        dw = (1.0 / n) * (x.T @ (pred - y))
        db = (1.0 / n) * np.sum(pred - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Миллер Игорь Владиславович, ПМ-31"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def fit(
        model: LinearRegression | LogisticRegression,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_iter: int,
        batch_size: int | None = None,
    ) -> None:
        if batch_size is None:
            for _ in range(n_iter):
                dw, db = model.grad(x, y)
                model.weights -= lr * dw
                model.bias -= lr * db
        else:
            for _ in range(n_iter):
                for i in range(0, len(x), batch_size):
                    x_batch = x[i : i + batch_size]
                    y_batch = y[i : i + batch_size]

                    dw, db = model.grad(x_batch, y_batch)
                    model.weights -= lr * dw
                    model.bias -= lr * db

    @staticmethod
    def get_iris_hyperparameters() -> dict[str, int | float]:
        return {"lr": 0.03, "batch_size": 1}
