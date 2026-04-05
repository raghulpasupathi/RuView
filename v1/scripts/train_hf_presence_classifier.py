"""
Train a lightweight RSSI classifier from JSONL logs.

Usage:
  python -m v1.scripts.train_hf_presence_classifier --input ./rssi_log.jsonl --output ./trained_models/hf_presence.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURE_KEYS = [
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

LABEL_TO_ID = {"absent": 0, "present_still": 1, "active": 2}


def row_to_x(row: dict) -> List[float]:
    f = row.get("features") or {}
    return [float(f.get(k, 0.0)) for k in FEATURE_KEYS]


def row_to_y(row: dict) -> int:
    cl = row.get("classification") or {}
    motion = str(cl.get("motion_level", "absent"))
    return LABEL_TO_ID.get(motion, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()

    inp = Path(ns.input)
    out = Path(ns.output)
    x, y = [], []
    with inp.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            if row.get("type") == "sensing_update":
                x.append(row_to_x(row))
                y.append(row_to_y(row))
            elif "mean_rssi" in row:
                # fallback for compact log-jsonl format
                fx = [float(row.get("mean_rssi", 0.0)), float(row.get("variance", 0.0)), 0.0,
                      float(row.get("motion_band_power", 0.0)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                x.append(fx)
                y.append(2 if row.get("presence") else 0)

    if len(x) < 20:
        raise SystemExit(f"Not enough samples to train: {len(x)}")

    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.int64)
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, Y)

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(out))
    print(f"saved model -> {out}")
    print(f"samples={len(X)} features={X.shape[1]}")


if __name__ == "__main__":
    main()
