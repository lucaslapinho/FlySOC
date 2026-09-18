"""Experimental KC-to-MBON readout with auditable online feedback updates.

The update is an engineered three-factor-style rule (KC activity x output
error x configured reward). It is not a biophysical dopamine model and is not
connected to :class:`flysoc.memory.AssociativeMemory` or the main pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


DEFAULT_CLASSES = ("TRUE_POSITIVE", "FALSE_POSITIVE", "BENIGN")


class MBONInspiredReadout:
    """Incremental multiclass readout for prequential feedback experiments.

    ``reward_policy`` maps every output label to a finite signed multiplier.
    Positive values learn toward the supplied label, negative values learn
    away from it, and zero records the event without changing parameters. The
    default gives every class the same multiplier; it deliberately assigns no
    special operational value to a particular SOC verdict.
    """

    def __init__(
        self,
        kc_dim: int,
        classes: Sequence[str] = DEFAULT_CLASSES,
        *,
        reward_policy: Mapping[str, float] | None = None,
        learning_rate: float = 0.10,
        weight_decay: float = 0.0,
        max_abs_weight: float = 5.0,
        max_abs_bias: float = 5.0,
    ) -> None:
        if kc_dim < 1:
            raise ValueError("kc_dim must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(weight_decay) or not 0 <= weight_decay < 1:
            raise ValueError("weight_decay must be finite and in [0, 1)")
        if not np.isfinite(max_abs_weight) or max_abs_weight <= 0:
            raise ValueError("max_abs_weight must be finite and positive")
        if not np.isfinite(max_abs_bias) or max_abs_bias <= 0:
            raise ValueError("max_abs_bias must be finite and positive")

        class_list = [str(label) for label in classes]
        if len(class_list) < 2 or len(set(class_list)) != len(class_list):
            raise ValueError("At least two unique output classes are required")

        if reward_policy is None:
            policy = {label: 1.0 for label in class_list}
        else:
            policy = {str(label): float(reward) for label, reward in reward_policy.items()}
            missing = set(class_list) - set(policy)
            extra = set(policy) - set(class_list)
            if missing or extra:
                raise ValueError(
                    "reward_policy must define exactly the configured classes; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
        if not all(np.isfinite(reward) for reward in policy.values()):
            raise ValueError("reward_policy values must be finite")

        self.kc_dim = int(kc_dim)
        self.classes_ = np.asarray(class_list, dtype=object)
        self.reward_policy = policy
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_abs_weight = float(max_abs_weight)
        self.max_abs_bias = float(max_abs_bias)
        self.weights_ = np.zeros((self.kc_dim, len(self.classes_)), dtype=np.float32)
        self.bias_ = np.zeros(len(self.classes_), dtype=np.float32)
        self.updates_ = 0
        self.feedback_events_ = 0
        self.history_: list[dict[str, Any]] = []

    def _matrix(self, x: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
        matrix = sparse.csr_matrix(x, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.kc_dim:
            raise ValueError(f"Expected a 2D KC matrix with width {self.kc_dim}")
        if not np.isfinite(matrix.data).all():
            raise ValueError("KC activity contains non-finite values")
        return matrix

    def decision_function(self, x: sparse.spmatrix | np.ndarray) -> np.ndarray:
        matrix = self._matrix(x)
        return np.asarray(matrix @ self.weights_ + self.bias_, dtype=np.float32)

    def predict_proba(self, x: sparse.spmatrix | np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, x: sparse.spmatrix | np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def partial_fit(
        self,
        x: sparse.spmatrix | np.ndarray,
        y: Sequence[str],
        *,
        event_ids: Sequence[str | None] | None = None,
        notes: Sequence[str] | None = None,
    ) -> "MBONInspiredReadout":
        """Apply feedback one row at a time and append an audit event per row.

        For valid chronological evaluation, predict an event before passing its
        label here; only later events may benefit from the resulting update.
        """

        matrix = self._matrix(x)
        labels = [str(label) for label in y]
        row_count = matrix.shape[0]
        if len(labels) != row_count:
            raise ValueError("Label count does not match KC rows")
        if event_ids is None:
            event_values: list[str | None] = [None] * row_count
        else:
            event_values = [None if value is None else str(value) for value in event_ids]
            if len(event_values) != row_count:
                raise ValueError("event_ids count does not match KC rows")
        if notes is None:
            note_values = [""] * row_count
        else:
            note_values = [str(value) for value in notes]
            if len(note_values) != row_count:
                raise ValueError("notes count does not match KC rows")

        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        unknown = sorted(set(labels) - set(class_to_index))
        if unknown:
            raise ValueError(f"Unknown feedback labels: {unknown}")

        for row_index, label in enumerate(labels):
            row = matrix.getrow(row_index)
            probabilities_before = self.predict_proba(row)[0]
            reward = self.reward_policy[label]
            applied = row.nnz > 0 and reward != 0.0
            clipped_weights = 0
            clipped_biases = 0

            if applied:
                target = np.zeros(len(self.classes_), dtype=np.float32)
                target[class_to_index[label]] = 1.0
                error = target - probabilities_before
                scale = self.learning_rate * reward / max(1, row.nnz)
                if self.weight_decay:
                    self.weights_ *= 1.0 - self.weight_decay
                self.weights_[row.indices] += row.data[:, None] * (scale * error[None, :])
                self.bias_ += self.learning_rate * 0.05 * reward * error

                clipped_weights = int(np.count_nonzero(np.abs(self.weights_) > self.max_abs_weight))
                clipped_biases = int(np.count_nonzero(np.abs(self.bias_) > self.max_abs_bias))
                np.clip(
                    self.weights_, -self.max_abs_weight, self.max_abs_weight, out=self.weights_
                )
                np.clip(self.bias_, -self.max_abs_bias, self.max_abs_bias, out=self.bias_)
                self.updates_ += 1

            probabilities_after = self.predict_proba(row)[0]
            self.feedback_events_ += 1
            reason = "updated" if applied else (
                "empty_fingerprint" if row.nnz == 0 else "zero_reward"
            )
            self.history_.append(
                {
                    "sequence": self.feedback_events_,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_id": event_values[row_index],
                    "label": label,
                    "reward": reward,
                    "applied": applied,
                    "reason": reason,
                    "active_kcs": int(row.nnz),
                    "prediction_before": str(self.classes_[probabilities_before.argmax()]),
                    "probabilities_before": {
                        str(output): float(probability)
                        for output, probability in zip(self.classes_, probabilities_before)
                    },
                    "prediction_after": str(self.classes_[probabilities_after.argmax()]),
                    "probabilities_after": {
                        str(output): float(probability)
                        for output, probability in zip(self.classes_, probabilities_after)
                    },
                    "clipped_weights": clipped_weights,
                    "clipped_biases": clipped_biases,
                    "note": note_values[row_index],
                }
            )
        return self

    def feedback_history(self) -> list[dict[str, Any]]:
        """Return a detached copy of the accepted feedback-event history."""

        return deepcopy(self.history_)
