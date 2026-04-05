"""
Adaptive coarse detector for RSSI-only presence/motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from v1.src.sensing.classifier import MotionLevel, SensingResult
from v1.src.sensing.feature_extractor import RssiFeatures
from v1.src.sensing.rssi_collector import WifiSample


@dataclass
class CoarseDetectorConfig:
    enter_threshold: float = 0.58
    exit_threshold: float = 0.36
    enter_ticks: int = 4
    exit_ticks: int = 6
    baseline_alpha: float = 0.025
    score_alpha: float = 0.22
    near_field_delta_dbm: float = 4.5
    near_field_range_dbm: float = 6.0
    near_field_block_threshold: float = 0.35
    in_between_presence_threshold: float = 0.60


class AdaptiveCoarseDetector:
    def __init__(self, cfg: Optional[CoarseDetectorConfig] = None) -> None:
        self.cfg = cfg or CoarseDetectorConfig()
        self._baseline_mean: Optional[float] = None
        self._score = 0.0
        self._stable_presence = False
        self._run_enter = 0
        self._run_exit = 0

    def classify(
        self,
        features: RssiFeatures,
        samples: Optional[List[WifiSample]] = None,
        fallback_result: Optional[SensingResult] = None,
    ) -> tuple[SensingResult, Dict[str, float]]:
        mean_rssi = float(features.mean)
        if self._baseline_mean is None:
            self._baseline_mean = mean_rssi

        calm = features.motion_band_power < 0.03 and features.variance < 0.18
        if (not self._stable_presence and calm) or (self._stable_presence and features.variance < 0.1):
            self._baseline_mean = (
                (1.0 - self.cfg.baseline_alpha) * self._baseline_mean
                + self.cfg.baseline_alpha * mean_rssi
            )

        delta = abs(mean_rssi - self._baseline_mean)
        var_n = float(np.clip(features.variance / 0.8, 0.0, 1.0))
        mot_n = float(np.clip(features.motion_band_power / 0.12, 0.0, 1.0))
        cp_n = float(np.clip(features.n_change_points / 8.0, 0.0, 1.0))
        iqr_n = float(np.clip(features.iqr / 1.6, 0.0, 1.0))

        near_field_penalty = 0.0
        if delta > self.cfg.near_field_delta_dbm and features.range > self.cfg.near_field_range_dbm:
            p1 = np.clip((delta - self.cfg.near_field_delta_dbm) / 4.5, 0.0, 1.0)
            p2 = np.clip((features.range - self.cfg.near_field_range_dbm) / 5.0, 0.0, 1.0)
            near_field_penalty = float(0.5 * p1 + 0.5 * p2)

        raw_score = 0.45 * var_n + 0.35 * mot_n + 0.10 * cp_n + 0.10 * iqr_n - 0.40 * near_field_penalty
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        self._score = (1.0 - self.cfg.score_alpha) * self._score + self.cfg.score_alpha * raw_score

        if self._score >= self.cfg.enter_threshold:
            self._run_enter += 1
            self._run_exit = 0
            if not self._stable_presence and self._run_enter >= self.cfg.enter_ticks:
                self._stable_presence = True
        elif self._score <= self.cfg.exit_threshold:
            self._run_exit += 1
            self._run_enter = 0
            if self._stable_presence and self._run_exit >= self.cfg.exit_ticks:
                self._stable_presence = False
        else:
            self._run_enter = 0
            self._run_exit = 0

        in_between_score = float(np.clip(0.55 * self._score + 0.45 * (1.0 - near_field_penalty), 0.0, 1.0))
        between_gate = (
            near_field_penalty <= self.cfg.near_field_block_threshold
            and in_between_score >= self.cfg.in_between_presence_threshold
        )
        # Keep between_gate as diagnostics only. Hard-gating presence here made
        # true in-between standing cases disappear.
        final_presence = self._stable_presence

        if not final_presence:
            level = MotionLevel.ABSENT
        elif mot_n >= 0.55:
            level = MotionLevel.ACTIVE
        else:
            level = MotionLevel.PRESENT_STILL

        conf = float(np.clip(abs(self._score - 0.5) * 2.0, 0.0, 1.0))
        if fallback_result is not None:
            conf = max(conf, float(fallback_result.confidence))
        result = SensingResult(
            motion_level=level,
            confidence=conf,
            presence_detected=final_presence,
            rssi_variance=float(features.variance),
            motion_band_energy=float(features.motion_band_power),
            breathing_band_energy=float(features.breathing_band_power),
            n_change_points=int(features.n_change_points),
            details=(
                f"adaptive score={self._score:.3f}, baseline={self._baseline_mean:.2f}, "
                f"delta={delta:.2f}, near_field={near_field_penalty:.2f}, "
                f"in_between={in_between_score:.2f}, between_gate={between_gate}"
            ),
        )
        diag = {
            "occupancy_score": float(self._score),
            "baseline_rssi": float(self._baseline_mean),
            "delta_from_baseline": float(delta),
            "near_field_penalty": float(near_field_penalty),
            "in_between_score": float(in_between_score),
            "between_gate": float(1.0 if between_gate else 0.0),
            "var_n": var_n,
            "motion_n": mot_n,
        }
        return result, diag
