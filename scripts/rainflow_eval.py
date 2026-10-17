"""Offline fatigue evaluation with strict rainflow + Goodman + Miner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from damage_estimator import DamageEstimatorConfig, turning_points


@dataclass
class OfflineEvalConfig:
    """Configuration for offline rainflow-like evaluation."""

    sn_k: float = 1.0e10
    sn_m: float = 3.0
    goodman_sigma_u: float = 1.0e9
    use_goodman: bool = True
    rainflow_mode: str = "strict"  # strict|simplified

    def as_damage_cfg(self) -> DamageEstimatorConfig:
        return DamageEstimatorConfig(
            window_size=10,
            sn_k=self.sn_k,
            sn_m=self.sn_m,
            goodman_sigma_u=self.goodman_sigma_u,
            use_goodman=self.use_goodman,
        )


def simplified_rainflow_cycles(signal: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (amplitude, mean, count) using adjacent turning-point pairs."""
    tp = turning_points(signal)
    if tp.size < 2:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty
    amp = 0.5 * np.abs(np.diff(tp))
    mean = 0.5 * (tp[:-1] + tp[1:])
    # Adjacent turning-point pairs are treated as half-cycles.
    count = np.full_like(amp, 0.5, dtype=float)
    return amp, mean, count


def strict_rainflow_cycles(signal: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (amplitude, mean, count) using strict rainflow counting.

    This is a stack-based implementation of the classic 4-point logic
    equivalent to ASTM-style cycle extraction.
    """
    tp = turning_points(signal)
    if tp.size < 2:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty

    stack: list[float] = []
    amp: list[float] = []
    mean: list[float] = []
    count: list[float] = []

    for value in tp:
        stack.append(float(value))
        while len(stack) >= 3:
            r0 = abs(stack[-2] - stack[-3])
            r1 = abs(stack[-1] - stack[-2])
            if r1 + 1e-12 < r0:
                break

            m = 0.5 * (stack[-3] + stack[-2])
            a = 0.5 * r0
            if len(stack) == 3:
                # Half cycle at the boundary.
                amp.append(a)
                mean.append(m)
                count.append(0.5)
                del stack[0]
            else:
                # Full cycle in the interior.
                amp.append(a)
                mean.append(m)
                count.append(1.0)
                del stack[-3:-1]

    # Remaining adjacent segments are half cycles.
    for i in range(len(stack) - 1):
        r = abs(stack[i + 1] - stack[i])
        amp.append(0.5 * r)
        mean.append(0.5 * (stack[i + 1] + stack[i]))
        count.append(0.5)

    return np.asarray(amp, dtype=float), np.asarray(mean, dtype=float), np.asarray(count, dtype=float)


def damage_from_signal(signal: np.ndarray | list[float], cfg: OfflineEvalConfig) -> float:
    """Computes Miner damage for one load signal."""
    mode = str(cfg.rainflow_mode).lower()
    if mode == "strict":
        amp, mean, count = strict_rainflow_cycles(signal)
    elif mode == "simplified":
        amp, mean, count = simplified_rainflow_cycles(signal)
    else:
        raise ValueError(f"Unsupported rainflow_mode: {cfg.rainflow_mode}")
    if amp.size == 0:
        return 0.0
    d_cfg = cfg.as_damage_cfg()
    # Scale per-cycle damage by cycle multiplicity.
    unit_damage = (np.maximum(amp, d_cfg.eps) ** d_cfg.sn_m) / max(d_cfg.sn_k, d_cfg.eps)
    if d_cfg.use_goodman and d_cfg.goodman_sigma_u > 0:
        denom = np.maximum(1.0 - mean / d_cfg.goodman_sigma_u, 0.05)
        amp_corr = np.maximum(amp / denom, d_cfg.eps)
        unit_damage = (amp_corr**d_cfg.sn_m) / max(d_cfg.sn_k, d_cfg.eps)
    return float(np.sum(unit_damage * count))


def gini(values: np.ndarray) -> float:
    """Computes Gini coefficient for non-negative values."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    sorted_x = np.sort(x)
    n = sorted_x.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * sorted_x) / (n * np.sum(sorted_x))) - (n + 1.0) / n)


def evaluate_damage(
    *,
    torque: np.ndarray,
    thrust: np.ndarray,
    cfg: OfflineEvalConfig,
) -> dict[str, Any]:
    """Evaluates turbine-wise and farm-wise fatigue metrics."""
    t = np.asarray(torque, dtype=float)
    f = np.asarray(thrust, dtype=float)
    if t.ndim != 2 or f.ndim != 2:
        raise ValueError("torque and thrust must be 2D arrays: [time, turbine].")
    if t.shape != f.shape:
        raise ValueError("torque and thrust must have the same shape.")

    _, n_turbines = t.shape
    d_t = np.zeros(n_turbines, dtype=float)
    d_f = np.zeros(n_turbines, dtype=float)
    for i in range(n_turbines):
        d_t[i] = damage_from_signal(t[:, i], cfg)
        d_f[i] = damage_from_signal(f[:, i], cfg)

    d_total = d_t + d_f
    return {
        "per_turbine_torque": d_t.tolist(),
        "per_turbine_thrust": d_f.tolist(),
        "per_turbine_total": d_total.tolist(),
        "D_farm": float(np.sum(d_total)),
        "D_max": float(np.max(d_total) if d_total.size else 0.0),
        "variance": float(np.var(d_total) if d_total.size else 0.0),
        "gini": gini(d_total),
    }
