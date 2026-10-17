"""Wind-farm load fitting and power-dispatch experiment runner.

Stages:
  A) Offline fit (first split)
  B) Online rolling scheduling (1 Hz)
  C) Offline fatigue evaluation
  D) Baseline comparison summary
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from baselines import legacy_baseline, mean_baseline, proportional_baseline
from damage_estimator import DamageEstimatorConfig, OnlineDamageEstimator
from load_estimator import (
    LoadEstimatorM1,
    LoadEstimatorPiecewise,
    PiecewiseFitOptions,
    StateMapperLinear,
    StateMapperOptions,
)
from optimizer import OptimizerConfig, optimize_power_step
from rainflow_eval import OfflineEvalConfig, evaluate_damage


def load_yaml_like(path: Path) -> dict[str, Any]:
    """Load a mapping with PyYAML's safe loader; reject malformed configuration."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for section in ("io", "experiment", "online", "load_model", "state_mapper",
                    "model_eval", "damage", "offline_eval", "dry_run"):
        if section in data and not isinstance(data[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def snapshot_code(project_root: Path, out_dir: Path, args: argparse.Namespace) -> None:
    snapshot_dir = ensure_dir(out_dir / "00_code_snapshot")
    scripts_src = project_root / "scripts"
    scripts_dst = snapshot_dir / "scripts"
    if scripts_dst.exists():
        shutil.rmtree(scripts_dst)
    shutil.copytree(
        scripts_src,
        scripts_dst,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )

    for filename in ["run_pipeline.sh", "run_optimizer_only.sh"]:
        src = project_root / filename
        if src.exists():
            shutil.copy2(src, snapshot_dir / filename)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "args": vars(args),
    }
    (snapshot_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def synthetic_data(*, total_steps: int, n_turbines: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(total_steps, dtype=float)[:, None]
    idx = np.arange(n_turbines, dtype=float)[None, :]
    v = 8.5 + 2.0 * np.sin(2.0 * np.pi * t / 120.0 + idx * 0.05) + rng.normal(
        0.0, 0.35, size=(total_steps, n_turbines)
    )
    v = np.clip(v, 3.0, 22.0)
    pref = np.clip(0.42 * v + 0.9 + 0.35 * np.sin(t / 33.0 + idx * 0.11), 0.0, 5.0)
    p_out = pref.copy()
    pitch = np.clip(2.2 + 0.12 * (20.0 - v) + rng.normal(0.0, 0.06, size=v.shape), 0.0, 20.0)
    omega_r = np.clip(8.0 + 0.6 * v + 0.2 * p_out + rng.normal(0.0, 0.18, size=v.shape), 0.0, None)
    omega_f = np.clip(70.0 + 6.0 * omega_r + rng.normal(0.0, 0.8, size=v.shape), 0.0, None)
    tshaft = np.clip(
        1.2 * p_out * v + 0.03 * (v**3) + 0.20 * omega_r + rng.normal(0.0, 0.8, size=v.shape),
        0.0,
        None,
    )
    ft = np.clip(0.25 * (v**2) + 0.6 * p_out + 0.03 * pitch + rng.normal(0.0, 0.6, size=v.shape), 0.0, None)
    return {
        "v": v,
        "pref": pref,
        "p_out": p_out,
        "tshaft": tshaft,
        "ft": ft,
        "pitch": pitch,
        "omega_r": omega_r,
        "omega_f": omega_f,
        "p_grid": np.sum(pref, axis=1),
    }


def load_npz_data(path: Path, *, farm: str = "") -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    with np.load(path, allow_pickle=False) as npz:
        data_all = {k: np.asarray(npz[k], dtype=float) for k in npz.files}

    farm_tag = str(farm).strip().lower()
    if farm_tag:
        prefix = f"{farm_tag}_"
        selected = {k[len(prefix) :]: v for k, v in data_all.items() if k.lower().startswith(prefix)}
        if not selected:
            raise KeyError(
                f"Farm '{farm}' not found in {path}. "
                f"Expected keys like '{farm_tag}_v', '{farm_tag}_pref', ..."
            )
        data = selected
    else:
        data = data_all
    required = {"v", "pref", "tshaft", "ft"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"NPZ is missing measured channels: {', '.join(sorted(missing))}")
    shape = data["pref"].shape
    if len(shape) != 2 or shape[0] < 12 or shape[1] == 0:
        raise ValueError("NPZ turbine arrays must have shape [time >= 12, turbines >= 1].")
    if "p_grid" not in data:
        data["p_grid"] = np.sum(data["pref"], axis=1)
    if "p_out" not in data:
        if "pout" in data:
            data["p_out"] = data["pout"].copy()
        else:
            raise ValueError("NPZ must contain measured power: p_out or pout.")
    for key in ("v", "pref", "p_out", "tshaft", "ft", "pitch", "omega_r", "omega_f", "p_grid"):
        if key not in data:
            continue
        expected = (shape[0],) if key == "p_grid" else shape
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"NPZ channel '{key}' must have shape {expected} and finite values.")
    return data


def parse_strategies(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = [x.strip() for x in str(raw).split(",")]
    out: list[str] = []
    allowed = {"mean", "proportional", "optimized", "optimized_legacy"}
    for item in items:
        if item:
            if item not in allowed:
                raise ValueError(f"Unsupported strategy: {item}")
            if item in out:
                raise ValueError(f"Duplicate strategy: {item}")
            out.append(item)
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _state_metrics(
    *,
    pitch_true: np.ndarray,
    pitch_pred: np.ndarray,
    omega_r_true: np.ndarray,
    omega_r_pred: np.ndarray,
    omega_f_true: np.ndarray,
    omega_f_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "pitch_rmse": _rmse(pitch_true, pitch_pred),
        "pitch_r2": _r2(pitch_true, pitch_pred),
        "omega_r_rmse": _rmse(omega_r_true, omega_r_pred),
        "omega_r_r2": _r2(omega_r_true, omega_r_pred),
        "omega_f_rmse": _rmse(omega_f_true, omega_f_pred),
        "omega_f_r2": _r2(omega_f_true, omega_f_pred),
    }


def _load_metrics(
    *,
    torque_true: np.ndarray,
    torque_pred: np.ndarray,
    thrust_true: np.ndarray,
    thrust_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "torque_rmse": _rmse(torque_true, torque_pred),
        "torque_r2": _r2(torque_true, torque_pred),
        "thrust_rmse": _rmse(thrust_true, thrust_pred),
        "thrust_r2": _r2(thrust_true, thrust_pred),
    }


def run_online_strategy(
    *,
    strategy: str,
    v_eval: np.ndarray,
    pref_eval: np.ndarray,
    p_grid_eval: np.ndarray,
    estimator: Any,
    opt_cfg: OptimizerConfig,
    damage_cfg: DamageEstimatorConfig,
    state_mapper: StateMapperLinear | None = None,
    use_historical_states: bool = False,
    pitch_eval: np.ndarray | None = None,
    omega_r_eval: np.ndarray | None = None,
    omega_f_eval: np.ndarray | None = None,
    v_prev_init: np.ndarray | None = None,
) -> dict[str, Any]:
    t_eval, n_turbines = pref_eval.shape
    p_sched = np.zeros((t_eval, n_turbines), dtype=float)
    torque_hat = np.zeros_like(p_sched)
    thrust_hat = np.zeros_like(p_sched)
    damage_step = np.zeros_like(p_sched)
    solve_ms = np.zeros(t_eval, dtype=float)

    damage_est = OnlineDamageEstimator(n_turbines=n_turbines, config=damage_cfg)
    p_prev = np.full(n_turbines, p_grid_eval[0] / float(n_turbines), dtype=float)
    first_prev = p_prev.copy()
    v_prev = v_prev_init.copy() if v_prev_init is not None else v_eval[0].copy()
    objective_proxy = np.zeros(t_eval, dtype=float)

    for k in range(t_eval):
        tic = time.perf_counter()
        p_grid_t = float(p_grid_eval[k])
        v_t = v_eval[k]
        pref_t = pref_eval[k]

        if strategy == "mean":
            p_t = mean_baseline(p_grid=p_grid_t, p_prev=p_prev, cfg=opt_cfg)
            info = {"objective_proxy": 0.0}
        elif strategy == "proportional":
            p_t = proportional_baseline(
                wind_speed=v_t,
                p_grid=p_grid_t,
                p_prev=p_prev,
                cfg=opt_cfg,
            )
            info = {"objective_proxy": 0.0}
        elif strategy == "optimized_legacy":
            p_t = legacy_baseline(
                pref=pref_t,
                wind_speed=v_t,
                p_grid=p_grid_t,
                p_prev=p_prev,
                cfg=opt_cfg,
            )
            info = {"objective_proxy": 0.0}
        elif strategy == "optimized":
            hint = damage_est.cumulative_damage + damage_est.last_step_damage + 1.0e-9
            p_t, info = optimize_power_step(
                p_prev=p_prev,
                p_grid=p_grid_t,
                damage_hint=hint,
                cfg=opt_cfg,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        if state_mapper is not None:
            pred_pitch, pred_omega_r, pred_omega_f = state_mapper.predict(v=v_t, p=p_t, p_prev=p_prev)
        elif use_historical_states and pitch_eval is not None and omega_r_eval is not None and omega_f_eval is not None:
            pred_pitch = pitch_eval[k]
            pred_omega_r = omega_r_eval[k]
            pred_omega_f = omega_f_eval[k]
        else:
            pred_pitch = None
            pred_omega_r = None
            pred_omega_f = None

        pred_t, pred_f = estimator.predict(
            v=v_t,
            p=p_t,
            p_prev=p_prev,
            pitch=pred_pitch,
            omega_r=pred_omega_r,
            omega_f=pred_omega_f,
            v_prev=v_prev,
        )
        d_t = damage_est.update(pred_t, pred_f)

        p_sched[k] = p_t
        torque_hat[k] = pred_t
        thrust_hat[k] = pred_f
        damage_step[k] = d_t
        objective_proxy[k] = float(info.get("objective_proxy", 0.0))
        solve_ms[k] = (time.perf_counter() - tic) * 1000.0
        v_prev = v_t
        p_prev = p_t

    mean_grid = p_grid_eval[:, None] / float(n_turbines)
    balance_mae = float(np.mean(np.abs(np.sum(p_sched, axis=1) - p_grid_eval)))
    if opt_cfg.max_dev > 0:
        max_dev_viol = int(np.sum(np.abs(p_sched - mean_grid) > (opt_cfg.max_dev + 1.0e-6)))
    else:
        max_dev_viol = 0

    p_prev_series = np.vstack([first_prev[None, :], p_sched[:-1]])
    if opt_cfg.ramp_max > 0:
        ramp_viol = int(np.sum(np.abs(p_sched - p_prev_series) > (opt_cfg.ramp_max + 1.0e-6)))
    else:
        ramp_viol = 0

    metrics = {
        "strategy": strategy,
        "solve_ms_mean": float(np.mean(solve_ms)),
        "solve_ms_max": float(np.max(solve_ms)),
        "solve_ms_p99": float(np.percentile(solve_ms, 99)),
        "balance_mae": balance_mae,
        "max_dev_violations": max_dev_viol,
        "ramp_violations": ramp_viol,
        "smooth_mean_abs_delta_p": float(np.mean(np.abs(p_sched - p_prev_series))),
        "objective_proxy_total": float(np.sum(objective_proxy)),
    }
    return {
        "schedule": p_sched,
        "torque_hat": torque_hat,
        "thrust_hat": thrust_hat,
        "damage_step": damage_step,
        "solve_ms": solve_ms,
        "online_metrics": metrics,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ISKE2026 experiment runner (plan-based).")
    parser.add_argument("--config", type=str, default="scripts/config.yaml")
    parser.add_argument("--data-npz", type=str, default="")
    parser.add_argument("--fit-data-npz", type=str, default="")
    parser.add_argument("--farm", type=str, default="")
    parser.add_argument("--fit-farm", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-start", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--strategies", type=str, default="")
    parser.add_argument("--skip-snapshot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_yaml_like(cfg_path)

    io_cfg = cfg.get("io", {})
    exp_cfg = cfg.get("experiment", {})
    online_cfg = cfg.get("online", {})
    load_model_cfg = cfg.get("load_model", {})
    state_mapper_cfg = cfg.get("state_mapper", {})
    model_eval_cfg = cfg.get("model_eval", {})
    damage_cfg_raw = cfg.get("damage", {})
    offline_cfg_raw = cfg.get("offline_eval", {})
    dry_cfg = cfg.get("dry_run", {})

    out_root = args.out_root or io_cfg.get("out_root", "outputs")
    run_name = args.run_name or f"run_{now_tag()}"
    if run_name in {".", ".."} or Path(run_name).name != run_name or "\\" in run_name or ":" in run_name:
        raise ValueError("Run name must be a single directory name without path separators.")
    out_dir = (project_root / out_root / run_name).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Run directory already exists: {out_dir}")

    strategies = (
        parse_strategies(args.strategies)
        if args.strategies
        else parse_strategies(exp_cfg.get("strategies", "mean,proportional,optimized_legacy,optimized"))
    )
    if not strategies:
        raise ValueError("No strategies specified.")

    rng_seed = int(args.seed)
    np.random.seed(rng_seed)

    cfg_data_npz = str(exp_cfg.get("data_npz", "")).strip()
    cfg_fit_data_npz = str(exp_cfg.get("fit_data_npz", "")).strip()
    cfg_farm = str(exp_cfg.get("farm", "")).strip()
    cfg_fit_farm = str(exp_cfg.get("fit_farm", "")).strip()

    data_npz_arg = args.data_npz or cfg_data_npz
    fit_data_npz_arg = args.fit_data_npz or cfg_fit_data_npz
    farm_arg = args.farm or cfg_farm
    fit_farm_arg = args.fit_farm or cfg_fit_farm

    if not args.dry_run and not data_npz_arg:
        raise ValueError("Specify a data NPZ file or use --dry-run for synthetic inputs.")
    if args.dry_run:
        eval_data = synthetic_data(
            total_steps=int(dry_cfg.get("total_steps", 2000)),
            n_turbines=int(dry_cfg.get("n_turbines", 100)),
            seed=rng_seed,
        )
        eval_data_mode = "dry_run"
        fit_data = eval_data
        fit_data_mode = eval_data_mode
    else:
        eval_data_path = Path(data_npz_arg)
        if not eval_data_path.is_absolute():
            eval_data_path = (project_root / eval_data_path).resolve()
        eval_data = load_npz_data(eval_data_path, farm=farm_arg)
        eval_data_mode = f"{eval_data_path}#{farm_arg}" if farm_arg else str(eval_data_path)

        fit_data_path_raw = fit_data_npz_arg or data_npz_arg
        fit_farm = fit_farm_arg or farm_arg
        if fit_data_npz_arg or fit_farm_arg:
            fit_data_path = Path(fit_data_path_raw)
            if not fit_data_path.is_absolute():
                fit_data_path = (project_root / fit_data_path).resolve()
            fit_data = load_npz_data(fit_data_path, farm=fit_farm)
            fit_data_mode = f"{fit_data_path}#{fit_farm}" if fit_farm else str(fit_data_path)
        else:
            fit_data = eval_data
            fit_data_mode = eval_data_mode

    # Fit dataset (Stage A)
    v_fit = np.asarray(fit_data["v"], dtype=float)
    pref_fit = np.asarray(fit_data["pref"], dtype=float)
    p_out_fit = np.asarray(fit_data["p_out"], dtype=float)
    tshaft_fit = np.asarray(fit_data["tshaft"], dtype=float)
    ft_fit = np.asarray(fit_data["ft"], dtype=float)
    p_grid_fit = np.asarray(fit_data["p_grid"], dtype=float)
    pitch_fit = np.asarray(fit_data["pitch"], dtype=float) if "pitch" in fit_data else None
    omega_r_fit = np.asarray(fit_data["omega_r"], dtype=float) if "omega_r" in fit_data else None
    omega_f_fit = np.asarray(fit_data["omega_f"], dtype=float) if "omega_f" in fit_data else None

    # Eval dataset (Stage B/C/D)
    v_eval_full = np.asarray(eval_data["v"], dtype=float)
    pref_eval_full = np.asarray(eval_data["pref"], dtype=float)
    p_out_eval = np.asarray(eval_data["p_out"], dtype=float)
    tshaft_eval_full = np.asarray(eval_data["tshaft"], dtype=float)
    ft_eval_full = np.asarray(eval_data["ft"], dtype=float)
    p_grid_eval_full = np.asarray(eval_data["p_grid"], dtype=float)
    pitch_eval_full = np.asarray(eval_data["pitch"], dtype=float) if "pitch" in eval_data else None
    omega_r_eval_full = np.asarray(eval_data["omega_r"], dtype=float) if "omega_r" in eval_data else None
    omega_f_eval_full = np.asarray(eval_data["omega_f"], dtype=float) if "omega_f" in eval_data else None

    total_steps_fit, n_turbines_fit = pref_fit.shape
    total_steps_eval, n_turbines_eval = pref_eval_full.shape
    if n_turbines_fit != n_turbines_eval:
        raise ValueError(
            f"Turbine count mismatch between fit/eval data: {n_turbines_fit} vs {n_turbines_eval}."
        )
    n_turbines = n_turbines_eval
    train_steps = int(exp_cfg.get("train_steps", 1000))
    eval_steps = int(exp_cfg.get("eval_steps", 1000))
    if not 12 <= train_steps <= total_steps_fit:
        raise ValueError("train_steps must be between 12 and the number of calibration samples.")
    eval_start_cfg = args.eval_start if args.eval_start is not None else exp_cfg.get("eval_start", None)
    auto_eval_start = eval_start_cfg is None or int(eval_start_cfg) < 0
    if auto_eval_start:
        eval_start = train_steps if fit_data is eval_data else 0
    else:
        eval_start = int(eval_start_cfg)
    eval_end = eval_start + eval_steps
    if eval_steps <= 0 or eval_start < 0 or eval_end > total_steps_eval:
        raise ValueError("Evaluation range must be nonempty and within the evaluation dataset.")
    if fit_data is eval_data and eval_start < train_steps:
        raise ValueError("Temporal calibration and evaluation slices must not overlap.")
    out_dir.mkdir(parents=True, exist_ok=False)

    # Stage A: offline fit.
    v_prev_fit = np.vstack([v_fit[0:1], v_fit[:-1]])
    p_prev_fit = np.vstack([p_out_fit[0:1], p_out_fit[:-1]])
    fit_ratio = float(model_eval_cfg.get("fit_ratio", 0.8))
    fit_ratio = min(max(fit_ratio, 0.5), 0.95)
    fit_steps = max(10, min(train_steps - 1, int(train_steps * fit_ratio)))
    fit_slice = slice(0, fit_steps)
    holdout_slice = slice(fit_steps, train_steps)
    use_historical_states = bool(online_cfg.get("use_historical_states", False))

    state_mapper: StateMapperLinear | None = None
    state_metrics: dict[str, Any] | None = None
    if (
        bool(state_mapper_cfg.get("enabled", True))
        and pitch_fit is not None
        and omega_r_fit is not None
        and omega_f_fit is not None
    ):
        state_mapper = StateMapperLinear(
            StateMapperOptions(
                reg_alpha=float(state_mapper_cfg.get("reg_alpha", 10.0)),
                normalize_features=bool(state_mapper_cfg.get("normalize_features", True)),
            )
        )
        state_mapper.fit(
            v=v_fit[fit_slice].reshape(-1),
            p=p_out_fit[fit_slice].reshape(-1),
            p_prev=p_prev_fit[fit_slice].reshape(-1),
            pitch=pitch_fit[fit_slice].reshape(-1),
            omega_r=omega_r_fit[fit_slice].reshape(-1),
            omega_f=omega_f_fit[fit_slice].reshape(-1),
        )

        tr_pitch, tr_omega_r, tr_omega_f = state_mapper.predict(
            v=v_fit[fit_slice].reshape(-1),
            p=p_out_fit[fit_slice].reshape(-1),
            p_prev=p_prev_fit[fit_slice].reshape(-1),
        )
        te_pitch, te_omega_r, te_omega_f = state_mapper.predict(
            v=v_fit[holdout_slice].reshape(-1),
            p=p_out_fit[holdout_slice].reshape(-1),
            p_prev=p_prev_fit[holdout_slice].reshape(-1),
        )
        state_metrics = {
            "train": _state_metrics(
                pitch_true=pitch_fit[fit_slice].reshape(-1),
                pitch_pred=tr_pitch,
                omega_r_true=omega_r_fit[fit_slice].reshape(-1),
                omega_r_pred=tr_omega_r,
                omega_f_true=omega_f_fit[fit_slice].reshape(-1),
                omega_f_pred=tr_omega_f,
            ),
            "test": _state_metrics(
                pitch_true=pitch_fit[holdout_slice].reshape(-1),
                pitch_pred=te_pitch,
                omega_r_true=omega_r_fit[holdout_slice].reshape(-1),
                omega_r_pred=te_omega_r,
                omega_f_true=omega_f_fit[holdout_slice].reshape(-1),
                omega_f_pred=te_omega_f,
            ),
        }

    load_model_type = str(load_model_cfg.get("type", "piecewise_linear")).lower()
    is_piecewise = load_model_type in {"piecewise", "piecewise_linear", "reference"}
    if is_piecewise:
        est_options = PiecewiseFitOptions(
            feature_mode=str(load_model_cfg.get("feature_mode", "auto")),
            reg_alpha=float(load_model_cfg.get("reg_alpha", 25.0)),
            normalize_features=bool(load_model_cfg.get("normalize_features", True)),
            quantiles=tuple(load_model_cfg.get("quantiles", [0.0, 0.33, 0.66, 1.0])),
        )
        estimator_eval = LoadEstimatorPiecewise(est_options)
        estimator = LoadEstimatorPiecewise(est_options)
        model_artifact_name = "load_estimator_piecewise.json"
    else:
        estimator_eval = LoadEstimatorM1()
        estimator = LoadEstimatorM1()
        model_artifact_name = "load_estimator_m1.json"

    def _prepare_fit_states(slc: slice) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if state_mapper is not None:
            ps, ors, ofs = state_mapper.predict(
                v=v_fit[slc].reshape(-1),
                p=p_out_fit[slc].reshape(-1),
                p_prev=p_prev_fit[slc].reshape(-1),
            )
            return ps, ors, ofs
        if use_historical_states and pitch_fit is not None and omega_r_fit is not None and omega_f_fit is not None:
            return pitch_fit[slc].reshape(-1), omega_r_fit[slc].reshape(-1), omega_f_fit[slc].reshape(-1)
        return None, None, None

    fit_pitch, fit_omega_r, fit_omega_f = _prepare_fit_states(fit_slice)
    hold_pitch, hold_omega_r, hold_omega_f = _prepare_fit_states(holdout_slice)

    estimator_eval.fit(
        v=v_fit[fit_slice].reshape(-1),
        p=p_out_fit[fit_slice].reshape(-1),
        p_prev=p_prev_fit[fit_slice].reshape(-1),
        tshaft=tshaft_fit[fit_slice].reshape(-1),
        ft=ft_fit[fit_slice].reshape(-1),
        pitch=fit_pitch,
        omega_r=fit_omega_r,
        omega_f=fit_omega_f,
        v_prev=v_prev_fit[fit_slice].reshape(-1),
    )
    pred_fit_t, pred_fit_f = estimator_eval.predict(
        v=v_fit[fit_slice].reshape(-1),
        p=p_out_fit[fit_slice].reshape(-1),
        p_prev=p_prev_fit[fit_slice].reshape(-1),
        pitch=fit_pitch,
        omega_r=fit_omega_r,
        omega_f=fit_omega_f,
        v_prev=v_prev_fit[fit_slice].reshape(-1),
    )
    pred_hold_t, pred_hold_f = estimator_eval.predict(
        v=v_fit[holdout_slice].reshape(-1),
        p=p_out_fit[holdout_slice].reshape(-1),
        p_prev=p_prev_fit[holdout_slice].reshape(-1),
        pitch=hold_pitch,
        omega_r=hold_omega_r,
        omega_f=hold_omega_f,
        v_prev=v_prev_fit[holdout_slice].reshape(-1),
    )

    fit_metrics = {
        "train": _load_metrics(
            torque_true=tshaft_fit[fit_slice].reshape(-1),
            torque_pred=pred_fit_t,
            thrust_true=ft_fit[fit_slice].reshape(-1),
            thrust_pred=pred_fit_f,
        ),
        "test": _load_metrics(
            torque_true=tshaft_fit[holdout_slice].reshape(-1),
            torque_pred=pred_hold_t,
            thrust_true=ft_fit[holdout_slice].reshape(-1),
            thrust_pred=pred_hold_f,
        ),
        "split": {"fit_steps": int(fit_steps), "holdout_steps": int(train_steps - fit_steps)},
    }

    full_pitch, full_omega_r, full_omega_f = _prepare_fit_states(slice(0, train_steps))
    estimator.fit(
        v=v_fit[:train_steps].reshape(-1),
        p=p_out_fit[:train_steps].reshape(-1),
        p_prev=p_prev_fit[:train_steps].reshape(-1),
        tshaft=tshaft_fit[:train_steps].reshape(-1),
        ft=ft_fit[:train_steps].reshape(-1),
        pitch=full_pitch,
        omega_r=full_omega_r,
        omega_f=full_omega_f,
        v_prev=v_prev_fit[:train_steps].reshape(-1),
    )
    stage_a_dir = ensure_dir(out_dir / "01_offline_fit")
    estimator.save_json(stage_a_dir / model_artifact_name, metrics=fit_metrics)
    if state_mapper is not None:
        state_mapper.save_json(stage_a_dir / "state_mapper_linear.json", metrics=state_metrics)

    opt_cfg = OptimizerConfig(
        p_min=float(online_cfg.get("p_min", 0.0)),
        p_max=float(online_cfg.get("p_max", 5.0)),
        max_dev=float(online_cfg.get("max_dev", 1.0)),
        ramp_max=float(online_cfg.get("ramp_max", 0.5)),
        dual_iters=int(online_cfg.get("dual_iters", 15)),
        dual_step=float(online_cfg.get("dual_step", 0.01)),
        wd=float(online_cfg.get("wd", 0.6)),
        wf=float(online_cfg.get("wf", 0.3)),
        ws=float(online_cfg.get("ws", 0.1)),
        damage_gain=float(online_cfg.get("damage_gain", 0.25)),
    )
    damage_cfg = DamageEstimatorConfig(
        window_size=int(damage_cfg_raw.get("window_size", 10)),
        sn_k=float(damage_cfg_raw.get("sn_k", 1.0e10)),
        sn_m=float(damage_cfg_raw.get("sn_m", 3.0)),
        goodman_sigma_u=float(damage_cfg_raw.get("goodman_sigma_u", 1.0e9)),
        use_goodman=bool(damage_cfg_raw.get("use_goodman", True)),
    )
    offline_cfg = OfflineEvalConfig(
        sn_k=float(offline_cfg_raw.get("sn_k", damage_cfg.sn_k)),
        sn_m=float(offline_cfg_raw.get("sn_m", damage_cfg.sn_m)),
        goodman_sigma_u=float(offline_cfg_raw.get("goodman_sigma_u", damage_cfg.goodman_sigma_u)),
        use_goodman=bool(offline_cfg_raw.get("use_goodman", damage_cfg.use_goodman)),
        rainflow_mode=str(offline_cfg_raw.get("rainflow_mode", "strict")),
    )

    v_eval = v_eval_full[eval_start:eval_end]
    pref_eval = pref_eval_full[eval_start:eval_end]
    p_grid_eval = p_grid_eval_full[eval_start:eval_end]
    pitch_eval = pitch_eval_full[eval_start:eval_end] if pitch_eval_full is not None else None
    omega_r_eval = omega_r_eval_full[eval_start:eval_end] if omega_r_eval_full is not None else None
    omega_f_eval = omega_f_eval_full[eval_start:eval_end] if omega_f_eval_full is not None else None
    v_prev_eval_init = v_eval_full[eval_start - 1] if eval_start > 0 else v_eval[0]

    # Stage B + C + D.
    stage_bc_dir = ensure_dir(out_dir / "02_online_and_offline")
    final_rows: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {
        "config": {
            "data_mode": {
                "fit": fit_data_mode,
                "eval": eval_data_mode,
            },
            "total_steps_fit": int(total_steps_fit),
            "total_steps_eval": int(total_steps_eval),
            "n_turbines": int(n_turbines),
            "train_steps": int(train_steps),
            "eval_range": [int(eval_start), int(eval_end)],
            "strategies": strategies,
            "load_model": {
                "type": load_model_type,
                **({} if not isinstance(estimator, LoadEstimatorPiecewise) else asdict(estimator.options)),
            },
            "state_mapper": {
                "enabled": state_mapper is not None,
                "use_historical_states": use_historical_states,
                "reg_alpha": float(state_mapper_cfg.get("reg_alpha", 10.0)),
                "normalize_features": bool(state_mapper_cfg.get("normalize_features", True)),
            },
            "optimizer": asdict(opt_cfg),
            "damage_online": asdict(damage_cfg),
            "damage_offline": asdict(offline_cfg),
        },
        "fit_metrics": fit_metrics,
        "state_mapper_metrics": state_metrics,
        "results": {},
    }

    for strategy in strategies:
        result = run_online_strategy(
            strategy=strategy,
            v_eval=v_eval,
            pref_eval=pref_eval,
            p_grid_eval=p_grid_eval,
            estimator=estimator,
            opt_cfg=opt_cfg,
            damage_cfg=damage_cfg,
            state_mapper=state_mapper,
            use_historical_states=use_historical_states,
            pitch_eval=pitch_eval,
            omega_r_eval=omega_r_eval,
            omega_f_eval=omega_f_eval,
            v_prev_init=v_prev_eval_init,
        )
        schedule_path = stage_bc_dir / f"schedule_{strategy}.npz"
        np.savez_compressed(
            schedule_path,
            p_sched=result["schedule"],
            torque_hat=result["torque_hat"],
            thrust_hat=result["thrust_hat"],
            damage_step=result["damage_step"],
            solve_ms=result["solve_ms"],
        )
        offline = evaluate_damage(
            torque=result["torque_hat"],
            thrust=result["thrust_hat"],
            cfg=offline_cfg,
        )
        metrics = {
            **result["online_metrics"],
            "D_farm": float(offline["D_farm"]),
            "D_max": float(offline["D_max"]),
            "damage_variance": float(offline["variance"]),
            "damage_gini": float(offline["gini"]),
        }
        (stage_bc_dir / f"online_metrics_{strategy}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_metrics["results"][strategy] = {
            "online_metrics": result["online_metrics"],
            "offline_metrics": offline,
            "schedule_path": str(schedule_path),
        }
        final_rows.append(
            {
                "strategy": strategy,
                "D_farm": metrics["D_farm"],
                "D_max": metrics["D_max"],
                "damage_variance": metrics["damage_variance"],
                "damage_gini": metrics["damage_gini"],
                "solve_ms_p99": metrics["solve_ms_p99"],
                "balance_mae": metrics["balance_mae"],
                "max_dev_violations": metrics["max_dev_violations"],
                "ramp_violations": metrics["ramp_violations"],
            }
        )

    final_table_path = out_dir / "final_table.csv"
    write_table(final_table_path, final_rows)
    final_metrics_path = out_dir / "final_metrics.json"
    final_metrics_path.write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_snapshot and bool(io_cfg.get("snapshot_code", True)):
        snapshot_code(project_root=project_root, out_dir=out_dir, args=args)

    print(f"[INFO] Run directory  : {out_dir}")
    print(f"[INFO] Final table    : {final_table_path}")
    print(f"[INFO] Final metrics  : {final_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
