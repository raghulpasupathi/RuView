"""
RSSI sensing server with WebSocket stream + dashboard HTTP endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from v1.src.sensing.classifier import MotionLevel, PresenceClassifier, SensingResult
from v1.src.sensing.feature_extractor import RssiFeatureExtractor, RssiFeatures
from v1.src.sensing.rssi_collector import WifiSample, create_collector
from v1.src.sensing.coarse_detector import AdaptiveCoarseDetector
from v1.src.sensing.hf_classifier import HfClassifier, HfClassifierConfig

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    tick_interval: float = 0.5
    window_seconds: float = 20.0
    interface: str = "wlan0"
    wifi_only: bool = False
    presence_variance: float = 0.3
    motion_energy: float = 0.06
    present_enter_ticks: int = 2
    absent_exit_ticks: int = 4
    rssi_ema_alpha: float = 0.25
    rssi_median_kernel: int = 3
    log_jsonl: Optional[Path] = None
    node_distance_m: float = 2.5
    hf_model_local: Optional[Path] = None
    hf_presence_threshold: float = 0.55
    hf_active_threshold: float = 0.60


class PresenceHysteresis:
    def __init__(self, enter_ticks: int, exit_ticks: int) -> None:
        self.enter_ticks = max(1, int(enter_ticks))
        self.exit_ticks = max(1, int(exit_ticks))
        self._stable = False
        self._enter = 0
        self._exit = 0

    def update(self, raw_presence: bool) -> bool:
        if raw_presence:
            self._enter += 1
            self._exit = 0
            if not self._stable and self._enter >= self.enter_ticks:
                self._stable = True
        else:
            self._exit += 1
            self._enter = 0
            if self._stable and self._exit >= self.exit_ticks:
                self._stable = False
        return self._stable


def smooth_rssi_samples_median(samples: List[WifiSample], kernel: int) -> List[WifiSample]:
    k = max(1, int(kernel))
    if k <= 1 or len(samples) < k:
        return samples
    values = [s.rssi_dbm for s in samples]
    half = k // 2
    out: List[WifiSample] = []
    for i, s in enumerate(samples):
        lo = max(0, i - half)
        hi = min(len(samples), i + half + 1)
        med = sorted(values[lo:hi])[len(values[lo:hi]) // 2]
        out.append(replace(s, rssi_dbm=float(med)))
    return out


def smooth_rssi_samples_ema(samples: List[WifiSample], alpha: float) -> List[WifiSample]:
    a = float(alpha)
    if a <= 0.0 or a >= 1.0 or len(samples) < 2:
        return samples
    out: List[WifiSample] = []
    ema = samples[0].rssi_dbm
    out.append(samples[0])
    for s in samples[1:]:
        ema = a * s.rssi_dbm + (1.0 - a) * ema
        out.append(replace(s, rssi_dbm=float(ema)))
    return out


def generate_signal_field(features: RssiFeatures, result: SensingResult, grid_size: int = 20) -> Dict:
    field = [[0.05 for _ in range(grid_size)] for _ in range(grid_size)]
    cx = grid_size // 2
    cz = grid_size // 2
    motion = max(0.0, min(1.0, features.motion_band_power / 6.0))
    vari = max(0.0, min(1.0, features.variance / 4.0))
    if result.presence_detected:
        px = cx + int(3 * math.sin(time.time() * 0.25))
        pz = cz + int(2 * math.cos(time.time() * 0.2))
        sigma = 2.0 + vari
        for z in range(grid_size):
            for x in range(grid_size):
                dx = x - px
                dz = z - pz
                blob = math.exp(-(dx * dx + dz * dz) / (2.0 * sigma * sigma))
                field[z][x] += 0.25 + 0.75 * blob * (0.4 + 0.6 * motion)
    else:
        for z in range(grid_size):
            for x in range(grid_size):
                d = math.sqrt((x - cx) ** 2 + (z - cz) ** 2)
                field[z][x] += max(0.0, 0.12 - 0.01 * d)
    flat: List[float] = []
    for row in field:
        for v in row:
            flat.append(max(0.0, min(1.0, v)))
    return {"grid_size": [grid_size, 1, grid_size], "values": flat}


class SensingServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.clients: Set = set()
        self.collector = create_collector(
            preferred="windows" if config.wifi_only else "auto",
            interface=config.interface,
            sample_rate_hz=2.0,
        )
        self.source = type(self.collector).__name__.replace("Collector", "").lower()
        self.interface_name = getattr(self.collector, "_interface", config.interface)
        self.extractor = RssiFeatureExtractor(window_seconds=config.window_seconds)
        self.classifier = PresenceClassifier(
            presence_variance_threshold=config.presence_variance,
            motion_energy_threshold=config.motion_energy,
        )
        self.hysteresis = PresenceHysteresis(config.present_enter_ticks, config.absent_exit_ticks)
        self.coarse_detector = AdaptiveCoarseDetector()
        self.hf_classifier: Optional[HfClassifier] = None
        if config.hf_model_local:
            try:
                self.hf_classifier = HfClassifier(
                    HfClassifierConfig(model_local=config.hf_model_local)
                )
                logger.info("HF classifier loaded from %s", config.hf_model_local)
            except Exception:
                logger.exception("Failed to load HF classifier model: %s", config.hf_model_local)
                self.hf_classifier = None
        self._running = False
        self._last_broadcast: Optional[dict] = None

    def _append_jsonl(self, payload: dict) -> None:
        if not self.config.log_jsonl:
            return
        try:
            self.config.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.config.log_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            logger.exception("Failed writing jsonl log")

    def _build_message(
        self,
        features: RssiFeatures,
        result_raw: SensingResult,
        result_display: SensingResult,
        raw_presence: bool,
        hf_probs: Optional[Dict[str, float]] = None,
        detector_diag: Optional[Dict[str, float]] = None,
    ) -> str:
        msg = {
            "type": "sensing_update",
            "timestamp": time.time(),
            "source": self.source,
            "interface": self.interface_name,
            "nodes": [{
                "node_id": 1,
                "rssi_dbm": features.mean,
                "position": [float(self.config.node_distance_m), 0.0, 0.0],
                "amplitude": [],
                "subcarrier_count": 0,
            }],
            "features": {
                "mean_rssi": features.mean,
                "variance": features.variance,
                "std": features.std,
                "motion_band_power": features.motion_band_power,
                "breathing_band_power": features.breathing_band_power,
                "dominant_freq_hz": features.dominant_freq_hz,
                "change_points": features.n_change_points,
                "spectral_power": features.total_spectral_power,
                "range": features.range,
                "iqr": features.iqr,
                "skewness": features.skewness,
                "kurtosis": features.kurtosis,
            },
            "classification": {
                "motion_level": result_raw.motion_level.value,
                "presence": result_display.presence_detected,
                "raw_presence": raw_presence,
                "confidence": round(float(result_display.confidence), 3),
                "hf_probs": hf_probs or {},
                "detector": detector_diag or {},
            },
            "signal_field": generate_signal_field(features, result_display),
        }
        return json.dumps(msg)

    def _apply_hf_classifier(self, features: RssiFeatures, rule_result: SensingResult) -> tuple[SensingResult, Dict[str, float]]:
        if self.hf_classifier is None or not self.hf_classifier.loaded:
            return rule_result, {}
        fmap = {
            "mean_rssi": features.mean,
            "variance": features.variance,
            "std": features.std,
            "motion_band_power": features.motion_band_power,
            "breathing_band_power": features.breathing_band_power,
            "dominant_freq_hz": features.dominant_freq_hz,
            "change_points": features.n_change_points,
            "spectral_power": features.total_spectral_power,
            "range": features.range,
            "iqr": features.iqr,
            "skewness": features.skewness,
            "kurtosis": features.kurtosis,
        }
        x = self.hf_classifier.feature_vector_from_map(fmap)
        probs = self.hf_classifier.predict_probabilities(x)
        if not probs:
            return rule_result, {}

        p_active = float(probs.get("active", 0.0))
        p_still = float(probs.get("present_still", 0.0))
        p_presence = p_active + p_still
        if p_presence < self.config.hf_presence_threshold:
            motion = MotionLevel.ABSENT
            presence = False
        elif p_active >= self.config.hf_active_threshold:
            motion = MotionLevel.ACTIVE
            presence = True
        else:
            motion = MotionLevel.PRESENT_STILL
            presence = True
        conf = max(float(rule_result.confidence), float(max(p_presence, p_active)))
        merged = replace(rule_result, motion_level=motion, presence_detected=presence, confidence=conf)
        return merged, probs

    async def _broadcast(self, message: str) -> None:
        if not self.clients:
            return
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    async def tick_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                n_needed = max(4, int(self.extractor.window_seconds * self.collector.sample_rate_hz))
                samples = self.collector.get_samples(n=n_needed)
                samples = smooth_rssi_samples_median(samples, self.config.rssi_median_kernel)
                samples = smooth_rssi_samples_ema(samples, self.config.rssi_ema_alpha)
                if len(samples) >= 4:
                    features = self.extractor.extract(samples)
                    rule_result = self.classifier.classify(features)
                    result_raw, hf_probs = self._apply_hf_classifier(features, rule_result)
                    adaptive_result, detector_diag = self.coarse_detector.classify(
                        features, samples=samples, fallback_result=result_raw
                    )
                    raw_pres = adaptive_result.presence_detected
                    stable = self.hysteresis.update(raw_pres)
                    display = replace(adaptive_result, presence_detected=stable)
                    if display.confidence < 0.80:
                        display = replace(display, motion_level=MotionLevel.ABSENT, presence_detected=False)

                    msg = self._build_message(
                        features,
                        adaptive_result,
                        display,
                        raw_pres,
                        hf_probs=hf_probs,
                        detector_diag=detector_diag,
                    )
                    await self._broadcast(msg)

                    self._last_broadcast = {
                        "ts": time.time(),
                        "source": self.source,
                        "motion_level": adaptive_result.motion_level.value,
                        "presence": display.presence_detected,
                        "raw_presence": raw_pres,
                        "confidence": float(display.confidence),
                        "mean_rssi": float(features.mean),
                        "variance": float(features.variance),
                        "motion_band_power": float(features.motion_band_power),
                        "rssi_window": [float(s.rssi_dbm) for s in samples[-40:]],
                    }
                    self._append_jsonl(self._last_broadcast)
                    logger.info(
                        "%s source=%s motion=%s presence=%s (raw=%s) conf=%.0f%% rssi_mean=%.1f var=%.4f",
                        time.strftime("%H:%M:%S"),
                        self.source,
                        adaptive_result.motion_level.value,
                        display.presence_detected,
                        raw_pres,
                        display.confidence * 100.0,
                        features.mean,
                        features.variance,
                    )
            except Exception:
                logger.exception("Error in sensing tick")
            await asyncio.sleep(self.config.tick_interval)

    def stop(self) -> None:
        self._running = False
        try:
            self.collector.stop()
        except Exception:
            pass


def _http_response(status: int, content: bytes, content_type: str = "text/plain; charset=utf-8") -> Tuple[int, List[Tuple[str, str]], bytes]:
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(content))),
        ("Cache-Control", "no-store"),
    ]
    return status, headers, content


def parse_args() -> ServerConfig:
    p = argparse.ArgumentParser(description="RuView sensing WebSocket + presence dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tick-interval", type=float, default=0.5)
    p.add_argument("--window-seconds", type=float, default=20.0)
    p.add_argument("--interface", default="wlan0")
    p.add_argument("--wifi-only", action="store_true")
    p.add_argument("--presence-variance", type=float, default=0.3)
    p.add_argument("--motion-energy", type=float, default=0.06)
    p.add_argument("--present-enter-ticks", type=int, default=2)
    p.add_argument("--absent-exit-ticks", type=int, default=4)
    p.add_argument("--rssi-ema-alpha", type=float, default=0.25)
    p.add_argument("--rssi-median-kernel", type=int, default=3)
    p.add_argument("--log-jsonl", type=str, default=None)
    p.add_argument("--hf-model-local", type=str, default=None)
    p.add_argument("--hf-presence-threshold", type=float, default=0.55)
    p.add_argument("--hf-active-threshold", type=float, default=0.60)
    p.add_argument("--node-distance-m", type=float, default=2.5)
    p.add_argument("--node-distance-ft", type=float, default=None)
    ns = p.parse_args()
    node_m = float(ns.node_distance_m)
    if ns.node_distance_ft is not None:
        node_m = float(ns.node_distance_ft) * 0.3048
    return ServerConfig(
        host=ns.host,
        port=ns.port,
        tick_interval=ns.tick_interval,
        window_seconds=ns.window_seconds,
        interface=ns.interface,
        wifi_only=ns.wifi_only,
        presence_variance=ns.presence_variance,
        motion_energy=ns.motion_energy,
        present_enter_ticks=ns.present_enter_ticks,
        absent_exit_ticks=ns.absent_exit_ticks,
        rssi_ema_alpha=ns.rssi_ema_alpha,
        rssi_median_kernel=ns.rssi_median_kernel,
        log_jsonl=Path(ns.log_jsonl) if ns.log_jsonl else None,
        hf_model_local=Path(ns.hf_model_local) if ns.hf_model_local else None,
        hf_presence_threshold=ns.hf_presence_threshold,
        hf_active_threshold=ns.hf_active_threshold,
        node_distance_m=node_m,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = parse_args()
    server = SensingServer(config)

    try:
        import websockets
    except Exception:
        print("ERROR: 'websockets' package is required. Install: pip install websockets")
        raise

    server.collector.start()

    async def handler(websocket, path):
        # websocket endpoint
        if path not in ("/ws/sensing", "/"):
            await websocket.close()
            return
        server.clients.add(websocket)
        logger.info("WS client connected: %s", websocket.remote_address)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            server.clients.discard(websocket)
            logger.info("WS client disconnected: %s", websocket.remote_address)

    async def process_request(path, request_headers):
        if path in ("/ws/sensing", "/"):
            # let websockets handle upgrade when requested
            return None
        if path == "/health":
            body = json.dumps({
                "ok": True,
                "source": server.source,
                "interface": server.interface_name,
                "node_distance_m": server.config.node_distance_m,
                "last": server._last_broadcast,
            }).encode("utf-8")
            return _http_response(200, body, "application/json; charset=utf-8")
        if path in ("/", "/presence-dashboard.html"):
            page = _ROOT / "ui" / "presence-dashboard.html"
            if page.exists():
                return _http_response(200, page.read_bytes(), "text/html; charset=utf-8")
            return _http_response(404, b"ui/presence-dashboard.html missing\n")
        return _http_response(404, b"not found\n")

    print()
    print(f"  HTTP  http://{config.host}:{config.port}/health")
    print(f"  Dashboard  http://{config.host}:{config.port}/presence-dashboard.html")
    print(f"  WebSocket  ws://{config.host}:{config.port}/ws/sensing  (or ws://{config.host}:{config.port}/)")
    print(f"  Source: {server.source} | Interface: {server.interface_name}")
    print(f"  HF classifier: {'enabled' if server.hf_classifier and server.hf_classifier.loaded else 'disabled'}")
    print(
        f"  Tick: {config.tick_interval}s | Window: {config.window_seconds}s | "
        f"Hysteresis enter={config.present_enter_ticks} exit={config.absent_exit_ticks}"
    )
    print(f"  Node distance (viz): {config.node_distance_m:.2f} m")
    print("  Ctrl+C to stop")
    print()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _shutdown(*_args: object) -> None:
        server.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    async def runner():
        tick = asyncio.create_task(server.tick_loop())
        async with websockets.serve(handler, config.host, config.port, process_request=process_request):
            await stop_event.wait()
        tick.cancel()
        try:
            await tick
        except asyncio.CancelledError:
            pass

    try:
        loop.run_until_complete(runner())
    finally:
        server.stop()


if __name__ == "__main__":
    main()
