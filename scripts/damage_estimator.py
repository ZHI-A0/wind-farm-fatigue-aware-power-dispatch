"""Online damage increment estimator with short sliding windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DamageEstimatorConfig:
    """Configuration for S-N and online window settings."""

    window_size: int = 10
    sn_k: float = 1.0e10
    sn_m: float = 3.0
    goodman_sigma_u: float = 1.0e9
    use_goodman: bool = True
    eps: float = 1.0e-12


def turning_points(signal: Iterable[float]) -> np.ndarray:
    """Returns turning points (peaks + valleys) from a 1D signal."""
    x = np.asarray(list(signal), dtype=float).reshape(-1)
    if x.size <= 2:
        return x

    # Remove consecutive duplicates to avoid zero-slope ambiguity.
    keep = [x[0]]
    for value in x[1:]:
        if abs(value - keep[-1]) > 1.0e-12:
            keep.append(value)
    x = np.asarray(keep, dtype=float)
    if x.size <= 2:
        return x

    pts = [x[0]]
    for i in range(1, x.size - 1):
        d0 = x[i] - x[i - 1]
        d1 = x[i + 1] - x[i]
        if d0 == 0.0:
            continue
        if d0 * d1 <= 0.0:
            pts.append(x[i])
    pts.append(x[-1])
    return np.asarray(pts, dtype=float)


def simplified_cycle_amplitudes(signal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Approximates cycle amplitudes/means from adjacent turning points."""
    tp = turning_points(signal)
    if tp.size < 2:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    amp = 0.5 * np.abs(np.diff(tp))
    mean = 0.5 * (tp[:-1] + tp[1:])
    return amp, mean


def sn_miner_damage(
    amplitudes: np.ndarray,
    means: np.ndarray,
    *,
    config: DamageEstimatorConfig,
) -> float:
    """Computes Miner increment for amplitudes based on S-N law."""
    amp = np.asarray(amplitudes, dtype=float).reshape(-1)
    mean = np.asarray(means, dtype=float).reshape(-1)
    if amp.size == 0:
        return 0.0
    if mean.size != amp.size:
        raise ValueError("amplitudes and means must have same length.")

    s = np.maximum(amp, config.eps)
    if config.use_goodman and config.goodman_sigma_u > 0:
        denom = 1.0 - mean / config.goodman_sigma_u
        denom = np.maximum(denom, 0.05)
        s = s / denom

    # Nf = k * S^{-m}  -> 1/Nf = S^m / k
    damage = np.sum((s**config.sn_m) / max(config.sn_k, config.eps))
    return float(max(damage, 0.0))


class OnlineDamageEstimator:
    """Short-window online damage increment estimator."""

    def __init__(self, n_turbines: int, config: DamageEstimatorConfig) -> None:
        if n_turbines <= 0:
            raise ValueError("n_turbines must be positive.")
        if config.window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.n_turbines = int(n_turbines)
        self.config = config
        self.torque_windows = [deque(maxlen=config.window_size) for _ in range(self.n_turbines)]
        self.thrust_windows = [deque(maxlen=config.window_size) for _ in range(self.n_turbines)]
        self.cumulative_damage = np.zeros(self.n_turbines, dtype=float)
        self.last_step_damage = np.zeros(self.n_turbines, dtype=float)

    def reset(self) -> None:
        """Resets all windows and damage states."""
        for q in self.torque_windows:
            q.clear()
        for q in self.thrust_windows:
            q.clear()
        self.cumulative_damage.fill(0.0)
        self.last_step_damage.fill(0.0)

    def _window_damage(self, values: deque[float]) -> float:
        amp, mean = simplified_cycle_amplitudes(values)
        return sn_miner_damage(amp, mean, config=self.config)

    def update(self, torque: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        """Updates windows and returns per-turbine step damage increment."""
        t = np.asarray(torque, dtype=float).reshape(-1)
        f = np.asarray(thrust, dtype=float).reshape(-1)
        if t.size != self.n_turbines or f.size != self.n_turbines:
            raise ValueError("torque and thrust must have shape (n_turbines,).")

        step = np.zeros(self.n_turbines, dtype=float)
        for i in range(self.n_turbines):
            self.torque_windows[i].append(float(t[i]))
            self.thrust_windows[i].append(float(f[i]))
            d_t = self._window_damage(self.torque_windows[i])
            d_f = self._window_damage(self.thrust_windows[i])
            step[i] = (d_t + d_f) / float(self.config.window_size)

        step = np.maximum(step, 0.0)
        self.cumulative_damage += step
        self.last_step_damage = step
        return step.copy()
