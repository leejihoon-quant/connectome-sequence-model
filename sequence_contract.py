"""Generic validation helpers for stateful sequence inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SequencePoint:
    sequence_id: int
    step: int
    needs_prediction: bool
    features: np.ndarray


class SequenceOrderGuard:
    """Validate ordered input and signal when recurrent state must be reset."""

    def __init__(self) -> None:
        self._sequence_id: int | None = None
        self._previous_step: int | None = None

    def accept(self, point: SequencePoint) -> bool:
        """Validate a point; return True when a new sequence starts."""
        if point.step < 0:
            raise ValueError("step must be non-negative")
        features = np.asarray(point.features)
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("features must be a finite one-dimensional vector")

        new_sequence = point.sequence_id != self._sequence_id
        if new_sequence:
            if point.step != 0:
                raise ValueError("a new sequence must start at step zero")
            self._sequence_id = point.sequence_id
            self._previous_step = None
        elif self._previous_step is not None and point.step != self._previous_step + 1:
            raise ValueError("steps must arrive in order without gaps")

        self._previous_step = point.step
        return new_sequence


def validate_prediction(value: Any, *, output_size: int) -> np.ndarray:
    """Convert a model result to a finite float32 vector of the expected size."""
    prediction = np.asarray(value, dtype=np.float32)
    if prediction.shape != (output_size,) or not np.isfinite(prediction).all():
        raise ValueError(f"prediction must contain {output_size} finite values")
    return prediction
