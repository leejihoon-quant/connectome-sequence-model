"""Memory-bounded, mergeable weighted Pearson correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightedMoments:
    """Sufficient statistics for merging weighted covariance blocks."""

    weight: float = 0.0
    mean_target: float = 0.0
    mean_prediction: float = 0.0
    target_m2: float = 0.0
    prediction_m2: float = 0.0
    cross_moment: float = 0.0
    rows: int = 0

    def add(self, target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        if target.ndim != 1 or prediction.shape != target.shape or weight.shape != target.shape:
            raise ValueError("target, prediction, and weight must be equal-length vectors")
        if not (np.isfinite(target).all() and np.isfinite(prediction).all() and np.isfinite(weight).all()):
            raise ValueError("inputs must be finite")
        if np.any(weight < 0):
            raise ValueError("weights must be non-negative")
        active = weight > 0
        if not active.any():
            return

        target, prediction, weight = target[active], prediction[active], weight[active]
        block_weight = float(weight.sum())
        block_mean_target = float(np.sum(weight * target) / block_weight)
        block_mean_prediction = float(np.sum(weight * prediction) / block_weight)
        centered_target = target - block_mean_target
        centered_prediction = prediction - block_mean_prediction
        block_target_m2 = float(np.sum(weight * centered_target**2))
        block_prediction_m2 = float(np.sum(weight * centered_prediction**2))
        block_cross = float(np.sum(weight * centered_target * centered_prediction))
        self._merge(block_weight, block_mean_target, block_mean_prediction,
                    block_target_m2, block_prediction_m2, block_cross, len(target))

    def _merge(
        self,
        weight: float,
        mean_target: float,
        mean_prediction: float,
        target_m2: float,
        prediction_m2: float,
        cross_moment: float,
        rows: int,
    ) -> None:
        if self.weight == 0:
            self.weight = weight
            self.mean_target = mean_target
            self.mean_prediction = mean_prediction
            self.target_m2 = target_m2
            self.prediction_m2 = prediction_m2
            self.cross_moment = cross_moment
            self.rows = rows
            return

        total = self.weight + weight
        target_delta = mean_target - self.mean_target
        prediction_delta = mean_prediction - self.mean_prediction
        correction = self.weight * weight / total
        self.target_m2 += target_m2 + target_delta**2 * correction
        self.prediction_m2 += prediction_m2 + prediction_delta**2 * correction
        self.cross_moment += cross_moment + target_delta * prediction_delta * correction
        self.mean_target += target_delta * weight / total
        self.mean_prediction += prediction_delta * weight / total
        self.weight = total
        self.rows += rows

    def correlation(self) -> float:
        if self.weight <= 0 or self.target_m2 <= 0 or self.prediction_m2 <= 0:
            return 0.0
        denominator = np.sqrt(self.target_m2 * self.prediction_m2)
        if not np.isfinite(denominator) or denominator == 0:
            return 0.0
        return float(np.clip(self.cross_moment / denominator, -1.0, 1.0))
