"""
Optional local classifier wrapper for RSSI features.

This module intentionally keeps dependencies lightweight:
- model artifact loaded via joblib
- expects sklearn-like ``predict_proba``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np


@dataclass
class HfClassifierConfig:
    model_local: Optional[Path] = None
    labels: tuple[str, ...] = ("absent", "present_still", "active")


class HfClassifier:
    def __init__(self, cfg: HfClassifierConfig) -> None:
        self.cfg = cfg
        self._model = None
        if cfg.model_local:
            self.load(cfg.model_local)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: Path) -> None:
        self._model = joblib.load(str(model_path))

    @staticmethod
    def feature_vector_from_map(features: Dict[str, float]) -> List[float]:
        keys = [
            "mean_rssi",
            "variance",
            "std",
            "motion_band_power",
            "breathing_band_power",
            "dominant_freq_hz",
            "change_points",
            "spectral_power",
            "range",
            "iqr",
            "skewness",
            "kurtosis",
        ]
        return [float(features.get(k, 0.0)) for k in keys]

    def predict_probabilities(self, features: Iterable[float]) -> Dict[str, float]:
        if self._model is None:
            return {}
        x = np.asarray(list(features), dtype=np.float64).reshape(1, -1)
        n_expected = getattr(self._model, "n_features_in_", None)
        if isinstance(n_expected, int) and n_expected > 0 and x.shape[1] != n_expected:
            # Backward compatibility: truncate/pad to model expectation.
            if x.shape[1] > n_expected:
                x = x[:, :n_expected]
            else:
                pad = np.zeros((1, n_expected - x.shape[1]), dtype=np.float64)
                x = np.concatenate([x, pad], axis=1)
        probs = self._model.predict_proba(x)[0]
        out: Dict[str, float] = {}
        for i, p in enumerate(probs):
            label = self.cfg.labels[i] if i < len(self.cfg.labels) else f"class_{i}"
            out[label] = float(p)
        return out

