"""Real-time optimizer for 1 Hz active power scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OptimizerConfig:
    """Solver configuration."""

    p_min: float = 0.0
    p_max: float = 5.0
    max_dev: float = 1.0
    ramp_max: float = 0.5
    dual_iters: int = 15
    dual_step: float = 0.01
    wd: float = 0.6
    wf: float = 0.3
    ws: float = 0.1
    damage_gain: float = 0.25


def _as_vector(name: str, x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be nonempty and contain only finite values.")
    return arr


def build_feasible_bounds(
    *,
    p_grid: float,
    p_prev: np.ndarray | list[float],
    cfg: OptimizerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds hard bounds under power/deviation/ramp constraints."""
    prev = _as_vector("p_prev", p_prev)
    values = [p_grid, cfg.p_min, cfg.p_max, cfg.max_dev, cfg.ramp_max,
              cfg.dual_step, cfg.wd, cfg.wf, cfg.ws, cfg.damage_gain]
    if not np.isfinite(values).all():
        raise ValueError("Power instruction and optimizer settings must be finite.")
    if cfg.p_min > cfg.p_max or cfg.max_dev < 0 or cfg.ramp_max < 0:
        raise ValueError("Invalid power, deviation, or ramp bounds.")
    n = prev.size
    mean = float(p_grid) / float(n)

    lower = np.full(n, cfg.p_min, dtype=float)
    upper = np.full(n, cfg.p_max, dtype=float)

    if cfg.max_dev > 0:
        lower = np.maximum(lower, mean - cfg.max_dev)
        upper = np.minimum(upper, mean + cfg.max_dev)
    if cfg.ramp_max > 0:
        lower = np.maximum(lower, prev - cfg.ramp_max)
        upper = np.minimum(upper, prev + cfg.ramp_max)

    if np.any(lower > upper + 1e-12):
        raise ValueError("Infeasible bounds: lower > upper for at least one turbine.")
    return lower, upper


def project_bounded_simplex(
    p_raw: np.ndarray | list[float],
    *,
    total: float,
    lower: np.ndarray | list[float],
    upper: np.ndarray | list[float],
    max_iter: int = 100,
) -> np.ndarray:
    """Projects vector onto {x | sum(x)=total, lower<=x<=upper}."""
    x = _as_vector("p_raw", p_raw)
    lb = _as_vector("lower", lower)
    ub = _as_vector("upper", upper)
    if not (x.size == lb.size == ub.size):
        raise ValueError("p_raw, lower and upper must have the same length.")
    if np.any(lb > ub):
        raise ValueError("lower must be <= upper element-wise.")

    total_f = float(total)
    if not np.isfinite(total_f) or max_iter <= 0:
        raise ValueError("Projection total must be finite and max_iter must be positive.")
    lb_sum = float(np.sum(lb))
    ub_sum = float(np.sum(ub))
    if total_f < lb_sum - 1e-9 or total_f > ub_sum + 1e-9:
        raise ValueError(
            f"Infeasible projection target: total={total_f:.6f}, "
            f"sum(lower)={lb_sum:.6f}, sum(upper)={ub_sum:.6f}."
        )

    lo = float(np.min(x - ub)) - 1.0
    hi = float(np.max(x - lb)) + 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        y = np.clip(x - mid, lb, ub)
        s = float(np.sum(y))
        if s > total_f:
            lo = mid
        else:
            hi = mid
    y = np.clip(x - 0.5 * (lo + hi), lb, ub)

    # Final numerical correction on free coordinates.
    residual = total_f - float(np.sum(y))
    if abs(residual) > 1e-10:
        for _ in range(5):
            if residual > 0:
                free = np.where(y < ub - 1e-12)[0]
                if free.size == 0:
                    break
                step = residual / float(free.size)
                y[free] = np.minimum(ub[free], y[free] + step)
            else:
                free = np.where(y > lb + 1e-12)[0]
                if free.size == 0:
                    break
                step = residual / float(free.size)
                y[free] = np.maximum(lb[free], y[free] + step)
            residual = total_f - float(np.sum(y))
            if abs(residual) <= 1e-10:
                break
    return y


def optimize_power_step(
    *,
    p_prev: np.ndarray | list[float],
    p_grid: float,
    damage_hint: np.ndarray | list[float],
    cfg: OptimizerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solves one scheduling step with dual update + hard projection."""
    prev = _as_vector("p_prev", p_prev)
    d = _as_vector("damage_hint", damage_hint)
    if d.size != prev.size:
        raise ValueError("damage_hint and p_prev must have same length.")

    lower, upper = build_feasible_bounds(p_grid=p_grid, p_prev=prev, cfg=cfg)
    n = prev.size
    mean = float(p_grid) / float(n)

    # Damage-guided target: lower power to higher damage turbines.
    d_center = d - np.mean(d)
    d_scale = float(np.std(d_center) + 1e-12)
    damage_push = (d_center / d_scale) * cfg.damage_gain
    raw = np.full(n, mean, dtype=float) - cfg.wd * damage_push
    # Smoothness keeps trajectory close to previous step.
    raw = (1.0 - min(max(cfg.ws, 0.0), 0.95)) * raw + min(max(cfg.ws, 0.0), 0.95) * prev

    lam = 0.0
    x = np.clip(raw, lower, upper)
    for _ in range(max(1, cfg.dual_iters)):
        x = np.clip(raw - lam, lower, upper)
        imbalance = float(np.sum(x) - p_grid)
        lam += cfg.dual_step * imbalance
    p = project_bounded_simplex(x, total=p_grid, lower=lower, upper=upper)

    step_abs = np.abs(p - prev)
    fairness_z = float(np.max(d))
    obj = float(cfg.wd * np.mean(d) + cfg.wf * fairness_z + cfg.ws * np.mean(step_abs))
    info = {
        "objective_proxy": obj,
        "fairness_z": fairness_z,
        "power_balance_error": float(np.sum(p) - p_grid),
        "smooth_l1": float(np.mean(step_abs)),
    }
    return p, info
