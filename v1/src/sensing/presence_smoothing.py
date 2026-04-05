"""
Presence smoothing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PresenceHysteresis:
    enter_ticks: int = 2
    exit_ticks: int = 4

    def __post_init__(self) -> None:
        self.enter_ticks = max(1, int(self.enter_ticks))
        self.exit_ticks = max(1, int(self.exit_ticks))
        self._stable = False
        self._enter_run = 0
        self._exit_run = 0

    @property
    def stable(self) -> bool:
        return self._stable

    def update(self, raw_presence: bool) -> bool:
        if raw_presence:
            self._enter_run += 1
            self._exit_run = 0
            if not self._stable and self._enter_run >= self.enter_ticks:
                self._stable = True
        else:
            self._exit_run += 1
            self._enter_run = 0
            if self._stable and self._exit_run >= self.exit_ticks:
                self._stable = False
        return self._stable

