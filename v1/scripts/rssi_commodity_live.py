"""
Quick local monitor for RSSI-only coarse presence.

Usage:
  python v1/scripts/rssi_commodity_live.py --interface "Wi-Fi 3"
"""

from __future__ import annotations

import argparse
import time

from v1.src.sensing.classifier import PresenceClassifier
from v1.src.sensing.feature_extractor import RssiFeatureExtractor
from v1.src.sensing.rssi_collector import create_collector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="wlan0")
    ap.add_argument("--window-seconds", type=float, default=20.0)
    ap.add_argument("--sample-rate-hz", type=float, default=2.0)
    ap.add_argument("--presence-variance", type=float, default=0.3)
    ap.add_argument("--motion-energy", type=float, default=0.06)
    ns = ap.parse_args()

    col = create_collector(preferred="auto", interface=ns.interface, sample_rate_hz=ns.sample_rate_hz)
    ext = RssiFeatureExtractor(window_seconds=ns.window_seconds)
    clf = PresenceClassifier(
        presence_variance_threshold=ns.presence_variance,
        motion_energy_threshold=ns.motion_energy,
    )
    col.start()
    print(f"collector={type(col).__name__} interface={getattr(col,'_interface',ns.interface)}")
    try:
        while True:
            n = max(4, int(ns.window_seconds * col.sample_rate_hz))
            samples = col.get_samples(n=n)
            if len(samples) >= 4:
                f = ext.extract(samples)
                r = clf.classify(f)
                print(
                    f"{time.strftime('%H:%M:%S')} mean={f.mean:.1f} var={f.variance:.3f} "
                    f"mot={f.motion_band_power:.3f} -> {r.motion_level.value} ({r.confidence:.0%})"
                )
            time.sleep(0.5)
    finally:
        col.stop()


if __name__ == "__main__":
    main()

