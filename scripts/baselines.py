"""Baseline strategies for power allocation."""

from __future__ import annotations

import numpy as np

from optimizer import OptimizerConfig, build_feasible_bounds, project_bounded_simplex


def _project(
    raw: np.ndarray,
    *,
    p_grid: float,
    p_prev: np.ndarray,
    cfg: OptimizerConfig,
) -> np.ndarray:
    lb, ub = build_feasible_bounds(p_grid=p_grid, p_prev=p_prev, cfg=cfg)
    return project_bounded_simplex(raw, total=p_grid, lower=lb, upper=ub)


def mean_baseline(*, p_grid: float, p_prev: np.ndarray, cfg: OptimizerConfig) -> np.ndarray:
    """Uniform allocation baseline."""
    n = p_prev.size
    raw = np.full(n, float(p_grid) / float(n), dtype=float)
    return _project(raw, p_grid=p_grid, p_prev=p_prev, cfg=cfg)


def proportional_baseline(
    *,
    wind_speed: np.ndarray,
    p_grid: float,
    p_prev: np.ndarray,
    cfg: OptimizerConfig,
    power: float = 3.0,
) -> np.ndarray:
    """Speed-proportional baseline: p_i ~ v_i^power."""
    v = np.asarray(wind_speed, dtype=float).reshape(-1)
    w = np.maximum(v, 1.0e-6) ** max(power, 0.0)
    raw = float(p_grid) * w / float(np.sum(w))
    return _project(raw, p_grid=p_grid, p_prev=p_prev, cfg=cfg)


def legacy_baseline(
    *,
    pref: np.ndarray,
    wind_speed: np.ndarray,
    p_grid: float,
    p_prev: np.ndarray,
    cfg: OptimizerConfig,
) -> np.ndarray:
    """Legacy-like baseline: smoothed blend of pref and speed-based ratio."""
    pref_vec = np.asarray(pref, dtype=float).reshape(-1)
    v = np.asarray(wind_speed, dtype=float).reshape(-1)
    if not (pref_vec.size == v.size == p_prev.size):
        raise ValueError("pref, wind_speed and p_prev must have the same length.")

    v_weight = np.maximum(v, 1.0e-6) ** 3
    v_alloc = float(p_grid) * v_weight / float(np.sum(v_weight))
    pref_weight = np.maximum(pref_vec, 1.0e-6)
    pref_alloc = float(p_grid) * pref_weight / float(np.sum(pref_weight))
    smooth_anchor = 0.60 * p_prev + 0.25 * pref_alloc + 0.15 * v_alloc
    return _project(smooth_anchor, p_grid=p_grid, p_prev=p_prev, cfg=cfg)
