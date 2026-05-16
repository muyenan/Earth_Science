from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        values = np.asarray(x, dtype=float)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return np.asarray(x, dtype=float) * self.scale_ + self.mean_


class MLPRegressor:
    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.006,
        epochs: int = 1500,
        l2: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.weights: dict[str, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x_scaled = self.x_scaler.fit(x).transform(x)
        y_scaled = self.y_scaler.fit(y).transform(y)

        rng = np.random.default_rng(self.random_state)
        rows, features = x_scaled.shape
        outputs = y_scaled.shape[1]
        w1 = rng.normal(0.0, np.sqrt(2.0 / max(1, features)), size=(features, self.hidden_units))
        b1 = np.zeros((1, self.hidden_units))
        w2 = rng.normal(0.0, np.sqrt(2.0 / max(1, self.hidden_units)), size=(self.hidden_units, outputs))
        w_skip = np.zeros((features, outputs))
        b2 = np.zeros((1, outputs))

        adam = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in {"w1": w1, "b1": b1, "w2": w2, "w_skip": w_skip, "b2": b2}.items()}
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        for epoch in range(1, self.epochs + 1):
            hidden_raw = x_scaled @ w1 + b1
            hidden = np.tanh(hidden_raw)
            pred = hidden @ w2 + x_scaled @ w_skip + b2
            error = pred - y_scaled

            grad_pred = 2.0 * error / rows
            grads = {
                "w2": hidden.T @ grad_pred + self.l2 * w2,
                "w_skip": x_scaled.T @ grad_pred + self.l2 * w_skip,
                "b2": grad_pred.sum(axis=0, keepdims=True),
            }
            grad_hidden = grad_pred @ w2.T
            grad_hidden_raw = grad_hidden * (1.0 - hidden**2)
            grads["w1"] = x_scaled.T @ grad_hidden_raw + self.l2 * w1
            grads["b1"] = grad_hidden_raw.sum(axis=0, keepdims=True)

            for name, param in (("w1", w1), ("b1", b1), ("w2", w2), ("w_skip", w_skip), ("b2", b2)):
                m, v = adam[name]
                grad = grads[name]
                m *= beta1
                m += (1.0 - beta1) * grad
                v *= beta2
                v += (1.0 - beta2) * (grad**2)
                m_hat = m / (1.0 - beta1**epoch)
                v_hat = v / (1.0 - beta2**epoch)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.weights = {"w1": w1, "b1": b1, "w2": w2, "w_skip": w_skip, "b2": b2}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("Model is not fitted.")
        x_scaled = self.x_scaler.transform(np.asarray(x, dtype=float))
        hidden = np.tanh(x_scaled @ self.weights["w1"] + self.weights["b1"])
        pred_scaled = hidden @ self.weights["w2"] + x_scaled @ self.weights["w_skip"] + self.weights["b2"]
        return self.y_scaler.inverse_transform(pred_scaled)


class RidgeRegressor:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale == 0] = 1.0
        self.y_mean = y.mean(axis=0, keepdims=True)
        x_scaled = (x - self.x_mean) / self.x_scale
        centered_y = y - self.y_mean
        penalty = self.alpha * np.eye(x_scaled.shape[1])
        self.weights = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ centered_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_scale is None or self.y_mean is None or self.weights is None:
            raise RuntimeError("Model is not fitted.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_scaled = (x - self.x_mean) / self.x_scale
        return x_scaled @ self.weights + self.y_mean
