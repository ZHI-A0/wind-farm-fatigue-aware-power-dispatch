"""Load estimators for shaft torque and tower thrust.

Available estimators:
1) LoadEstimatorM1: simple direct linear regression from (v, p, p_prev).
2) LoadEstimatorPiecewise: reference-style piecewise linear regression with
   optional state features (pitch, omega_r, omega_f, dv, dp).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES: dict[str, list[str]] = {
    "vp": ["v", "p", "vp", "bias"],
    "vp_states": ["v", "p", "pitch", "omega_r", "omega_f", "vp", "v2", "dv", "dp", "bias"],
}


def _as_1d(name: str, arr: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return x


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _infer_states(v: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch = np.clip(2.2 + 0.12 * (20.0 - v), 0.0, 20.0)
    omega_r = np.clip(8.0 + 0.6 * v + 0.2 * p, 0.0, None)
    omega_f = np.clip(70.0 + 6.0 * omega_r, 0.0, None)
    return pitch, omega_r, omega_f


def _build_features(
    *,
    v: np.ndarray,
    p: np.ndarray,
    p_prev: np.ndarray,
    mode: str,
    pitch: np.ndarray | None = None,
    omega_r: np.ndarray | None = None,
    omega_f: np.ndarray | None = None,
    v_prev: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "vp":
        return np.column_stack([v, p, v * p, np.ones_like(v)])

    if mode != "vp_states":
        raise ValueError(f"Unsupported feature mode: {mode}")

    if pitch is None or omega_r is None or omega_f is None:
        pitch, omega_r, omega_f = _infer_states(v, p)
    if v_prev is None:
        dv = np.zeros_like(v)
    else:
        dv = v - v_prev
    dp = p - p_prev
    return np.column_stack([v, p, pitch, omega_r, omega_f, v * p, v**2, dv, dp, np.ones_like(v)])


def _fit_linear_stable(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reg_alpha: float,
    normalize_features: bool,
    bias_idx: int = -1,
) -> np.ndarray:
    x_raw = np.asarray(x, dtype=float)
    y_raw = np.asarray(y, dtype=float)
    if x_raw.ndim != 2:
        raise ValueError(f"x must be 2D, got {x_raw.shape}")
    d = x_raw.shape[1]
    if d == 0:
        raise ValueError("x has zero feature dimension.")

    bidx = bias_idx if bias_idx >= 0 else (d + bias_idx)
    if not (0 <= bidx < d):
        raise ValueError(f"Invalid bias index: {bias_idx} for d={d}")

    cols = np.array([i for i in range(d) if i != bidx], dtype=int)
    x_fit = x_raw.copy()
    mean = np.zeros(d, dtype=float)
    std = np.ones(d, dtype=float)

    if normalize_features and cols.size > 0:
        mean_cols = np.mean(x_fit[:, cols], axis=0)
        std_cols = np.std(x_fit[:, cols], axis=0)
        std_cols = np.where(std_cols < 1e-6, 1.0, std_cols)
        x_fit[:, cols] = (x_fit[:, cols] - mean_cols) / std_cols
        mean[cols] = mean_cols
        std[cols] = std_cols

    xtx = x_fit.T @ x_fit
    if reg_alpha > 0:
        reg = np.eye(d, dtype=float) * float(reg_alpha)
        reg[bidx, bidx] = 0.0
        xtx = xtx + reg
    xty = x_fit.T @ y_raw

    try:
        coef_fit = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef_fit = np.linalg.pinv(xtx) @ xty

    if not normalize_features or cols.size == 0:
        return coef_fit

    coef_raw = coef_fit.copy()
    coef_raw[cols] = coef_fit[cols] / std[cols]
    coef_raw[bidx] = coef_fit[bidx] - float(np.sum(coef_fit[cols] * mean[cols] / std[cols]))
    return coef_raw


def _fit_piecewise_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reg_alpha: float,
    normalize_features: bool,
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    v = x[:, 0]
    bins = np.asarray(np.quantile(v, quantiles), dtype=float)
    for i in range(1, bins.size):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6

    coefs: list[list[float]] = []
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (v >= b0) & (v <= b1 if i == bins.size - 2 else v < b1)
        if int(np.sum(mask)) < max(20, x.shape[1] + 5):
            coef = _fit_linear_stable(
                x,
                y,
                reg_alpha=reg_alpha,
                normalize_features=normalize_features,
                bias_idx=-1,
            )
        else:
            coef = _fit_linear_stable(
                x[mask],
                y[mask],
                reg_alpha=reg_alpha,
                normalize_features=normalize_features,
                bias_idx=-1,
            )
        coefs.append(coef.tolist())
    return {"bins": bins.tolist(), "coefs": coefs}


def _predict_piecewise_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    v = x[:, 0]
    bins = list(model["bins"])
    coefs = [np.asarray(c, dtype=float) for c in model["coefs"]]
    y = np.zeros(x.shape[0], dtype=float)
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (v >= b0) & (v <= b1 if i == len(bins) - 2 else v < b1)
        if np.any(mask):
            y[mask] = x[mask] @ coefs[i]
    return y


@dataclass
class LoadFitMetrics:
    """Fit quality metrics for two targets."""

    torque_rmse: float
    torque_r2: float
    thrust_rmse: float
    thrust_r2: float


@dataclass
class StateFitMetrics:
    """Fit quality metrics for state mapping model."""

    pitch_rmse: float
    pitch_r2: float
    omega_r_rmse: float
    omega_r_r2: float
    omega_f_rmse: float
    omega_f_r2: float


@dataclass
class PiecewiseFitOptions:
    """Options for the piecewise load estimator."""

    feature_mode: str = "auto"  # auto|vp|vp_states
    reg_alpha: float = 25.0
    normalize_features: bool = True
    quantiles: tuple[float, ...] = (0.0, 0.33, 0.66, 1.0)


@dataclass
class StateMapperOptions:
    """Options for state mapper."""

    reg_alpha: float = 10.0
    normalize_features: bool = True


class StateMapperLinear:
    """Linear mapper from (v, p, p_prev) to (pitch, omega_r, omega_f)."""

    def __init__(self, options: StateMapperOptions | None = None) -> None:
        self.options = options or StateMapperOptions()
        self.pitch_coef: np.ndarray | None = None
        self.omega_r_coef: np.ndarray | None = None
        self.omega_f_coef: np.ndarray | None = None

    @staticmethod
    def _x(v: np.ndarray, p: np.ndarray, p_prev: np.ndarray) -> np.ndarray:
        dp = p - p_prev
        return np.column_stack([v, p, p_prev, v * p, v**2, np.abs(dp), np.ones_like(v)])

    def fit(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
        pitch: np.ndarray | list[float],
        omega_r: np.ndarray | list[float],
        omega_f: np.ndarray | list[float],
    ) -> StateFitMetrics:
        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        pitch_ = _as_1d("pitch", pitch)
        omega_r_ = _as_1d("omega_r", omega_r)
        omega_f_ = _as_1d("omega_f", omega_f)
        n = v_.size
        if not (p_.size == p_prev_.size == pitch_.size == omega_r_.size == omega_f_.size == n):
            raise ValueError("All inputs must have the same length.")

        x = self._x(v_, p_, p_prev_)
        self.pitch_coef = _fit_linear_stable(
            x,
            pitch_,
            reg_alpha=self.options.reg_alpha,
            normalize_features=self.options.normalize_features,
            bias_idx=-1,
        )
        self.omega_r_coef = _fit_linear_stable(
            x,
            omega_r_,
            reg_alpha=self.options.reg_alpha,
            normalize_features=self.options.normalize_features,
            bias_idx=-1,
        )
        self.omega_f_coef = _fit_linear_stable(
            x,
            omega_f_,
            reg_alpha=self.options.reg_alpha,
            normalize_features=self.options.normalize_features,
            bias_idx=-1,
        )

        pred_pitch, pred_omega_r, pred_omega_f = self.predict(v=v_, p=p_, p_prev=p_prev_)
        return StateFitMetrics(
            pitch_rmse=_rmse(pitch_, pred_pitch),
            pitch_r2=_r2(pitch_, pred_pitch),
            omega_r_rmse=_rmse(omega_r_, pred_omega_r),
            omega_r_r2=_r2(omega_r_, pred_omega_r),
            omega_f_rmse=_rmse(omega_f_, pred_omega_f),
            omega_f_r2=_r2(omega_f_, pred_omega_f),
        )

    def predict(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pitch_coef is None or self.omega_r_coef is None or self.omega_f_coef is None:
            raise RuntimeError("State mapper is not fitted yet.")
        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        if not (v_.size == p_.size == p_prev_.size):
            raise ValueError("v, p, p_prev must have same length.")
        x = self._x(v_, p_, p_prev_)
        pitch = np.clip(x @ self.pitch_coef, 0.0, 25.0)
        omega_r = np.clip(x @ self.omega_r_coef, 0.0, None)
        omega_f = np.clip(x @ self.omega_f_coef, 0.0, None)
        return pitch, omega_r, omega_f

    def to_dict(self) -> dict[str, Any]:
        if self.pitch_coef is None or self.omega_r_coef is None or self.omega_f_coef is None:
            raise RuntimeError("State mapper is not fitted yet.")
        return {
            "model": "state_mapper_linear",
            "fit_options": asdict(self.options),
            "pitch_coef": self.pitch_coef.tolist(),
            "omega_r_coef": self.omega_r_coef.tolist(),
            "omega_f_coef": self.omega_f_coef.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateMapperLinear":
        opts = payload.get("fit_options", {})
        model = cls(
            StateMapperOptions(
                reg_alpha=float(opts.get("reg_alpha", 10.0)),
                normalize_features=bool(opts.get("normalize_features", True)),
            )
        )
        model.pitch_coef = np.asarray(payload["pitch_coef"], dtype=float)
        model.omega_r_coef = np.asarray(payload["omega_r_coef"], dtype=float)
        model.omega_f_coef = np.asarray(payload["omega_f_coef"], dtype=float)
        return model

    def save_json(self, path: str | Path, *, metrics: StateFitMetrics | dict[str, Any] | None = None) -> Path:
        data = self.to_dict()
        if metrics is not None:
            if isinstance(metrics, dict):
                data["metrics"] = metrics
            else:
                data["metrics"] = asdict(metrics)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> "StateMapperLinear":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class LoadEstimatorM1:
    """Linear least-squares load estimator (M1)."""

    def __init__(self) -> None:
        self.torque_coef: np.ndarray | None = None
        self.thrust_coef: np.ndarray | None = None

    @staticmethod
    def _x_torque(v: np.ndarray, p: np.ndarray, p_prev: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(v), p, v, v**2, np.abs(p - p_prev)])

    @staticmethod
    def _x_thrust(v: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(v), v**2, p, v * p])

    def fit(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
        tshaft: np.ndarray | list[float],
        ft: np.ndarray | list[float],
        **_: Any,
    ) -> LoadFitMetrics:
        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        t_ = _as_1d("tshaft", tshaft)
        f_ = _as_1d("ft", ft)
        n = v_.size
        if not (p_.size == p_prev_.size == t_.size == f_.size == n):
            raise ValueError("v, p, p_prev, tshaft, ft must have the same length.")

        xt = self._x_torque(v_, p_, p_prev_)
        xf = self._x_thrust(v_, p_)
        self.torque_coef = np.linalg.lstsq(xt, t_, rcond=None)[0]
        self.thrust_coef = np.linalg.lstsq(xf, f_, rcond=None)[0]

        pred_t, pred_f = self.predict(v=v_, p=p_, p_prev=p_prev_)
        return LoadFitMetrics(
            torque_rmse=_rmse(t_, pred_t),
            torque_r2=_r2(t_, pred_t),
            thrust_rmse=_rmse(f_, pred_f),
            thrust_r2=_r2(f_, pred_f),
        )

    def predict(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
        **_: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.torque_coef is None or self.thrust_coef is None:
            raise RuntimeError("Model is not fitted yet.")
        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        if not (p_.size == p_prev_.size == v_.size):
            raise ValueError("v, p, p_prev must have the same length.")
        xt = self._x_torque(v_, p_, p_prev_)
        xf = self._x_thrust(v_, p_)
        return xt @ self.torque_coef, xf @ self.thrust_coef

    def to_dict(self) -> dict[str, Any]:
        if self.torque_coef is None or self.thrust_coef is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "model": "M1",
            "torque_coef": self.torque_coef.tolist(),
            "thrust_coef": self.thrust_coef.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoadEstimatorM1":
        model = cls()
        model.torque_coef = np.asarray(payload["torque_coef"], dtype=float)
        model.thrust_coef = np.asarray(payload["thrust_coef"], dtype=float)
        return model

    def save_json(self, path: str | Path, *, metrics: LoadFitMetrics | dict[str, Any] | None = None) -> Path:
        data: dict[str, Any] = self.to_dict()
        if metrics is not None:
            if isinstance(metrics, dict):
                data["metrics"] = metrics
            else:
                data["metrics"] = asdict(metrics)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> "LoadEstimatorM1":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class LoadEstimatorPiecewise:
    """Reference-style piecewise linear estimator with stable fitting."""

    def __init__(self, options: PiecewiseFitOptions | None = None) -> None:
        self.options = options or PiecewiseFitOptions()
        self.feature_mode: str | None = None
        self.torque_model: dict[str, Any] | None = None
        self.thrust_model: dict[str, Any] | None = None

    def _resolve_mode(
        self,
        *,
        pitch: np.ndarray | None,
        omega_r: np.ndarray | None,
        omega_f: np.ndarray | None,
    ) -> str:
        if self.options.feature_mode in {"vp", "vp_states"}:
            return self.options.feature_mode
        if pitch is not None and omega_r is not None and omega_f is not None:
            return "vp_states"
        return "vp"

    def fit(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
        tshaft: np.ndarray | list[float],
        ft: np.ndarray | list[float],
        pitch: np.ndarray | list[float] | None = None,
        omega_r: np.ndarray | list[float] | None = None,
        omega_f: np.ndarray | list[float] | None = None,
        v_prev: np.ndarray | list[float] | None = None,
        **_: Any,
    ) -> LoadFitMetrics:
        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        t_ = _as_1d("tshaft", tshaft)
        f_ = _as_1d("ft", ft)
        n = v_.size
        if not (p_.size == p_prev_.size == t_.size == f_.size == n):
            raise ValueError("v, p, p_prev, tshaft, ft must have the same length.")

        pitch_ = None if pitch is None else _as_1d("pitch", pitch)
        omega_r_ = None if omega_r is None else _as_1d("omega_r", omega_r)
        omega_f_ = None if omega_f is None else _as_1d("omega_f", omega_f)
        v_prev_ = None if v_prev is None else _as_1d("v_prev", v_prev)

        mode = self._resolve_mode(pitch=pitch_, omega_r=omega_r_, omega_f=omega_f_)
        x = _build_features(
            v=v_,
            p=p_,
            p_prev=p_prev_,
            mode=mode,
            pitch=pitch_,
            omega_r=omega_r_,
            omega_f=omega_f_,
            v_prev=v_prev_,
        )

        self.feature_mode = mode
        self.torque_model = _fit_piecewise_model(
            x,
            t_,
            reg_alpha=self.options.reg_alpha,
            normalize_features=self.options.normalize_features,
            quantiles=self.options.quantiles,
        )
        self.thrust_model = _fit_piecewise_model(
            x,
            f_,
            reg_alpha=self.options.reg_alpha,
            normalize_features=self.options.normalize_features,
            quantiles=self.options.quantiles,
        )

        pred_t, pred_f = self.predict(
            v=v_,
            p=p_,
            p_prev=p_prev_,
            pitch=pitch_,
            omega_r=omega_r_,
            omega_f=omega_f_,
            v_prev=v_prev_,
        )
        return LoadFitMetrics(
            torque_rmse=_rmse(t_, pred_t),
            torque_r2=_r2(t_, pred_t),
            thrust_rmse=_rmse(f_, pred_f),
            thrust_r2=_r2(f_, pred_f),
        )

    def predict(
        self,
        *,
        v: np.ndarray | list[float],
        p: np.ndarray | list[float],
        p_prev: np.ndarray | list[float],
        pitch: np.ndarray | list[float] | None = None,
        omega_r: np.ndarray | list[float] | None = None,
        omega_f: np.ndarray | list[float] | None = None,
        v_prev: np.ndarray | list[float] | None = None,
        **_: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.torque_model is None or self.thrust_model is None or self.feature_mode is None:
            raise RuntimeError("Model is not fitted yet.")

        v_ = _as_1d("v", v)
        p_ = _as_1d("p", p)
        p_prev_ = _as_1d("p_prev", p_prev)
        if not (v_.size == p_.size == p_prev_.size):
            raise ValueError("v, p, p_prev must have same length.")

        pitch_ = None if pitch is None else _as_1d("pitch", pitch)
        omega_r_ = None if omega_r is None else _as_1d("omega_r", omega_r)
        omega_f_ = None if omega_f is None else _as_1d("omega_f", omega_f)
        v_prev_ = None if v_prev is None else _as_1d("v_prev", v_prev)

        x = _build_features(
            v=v_,
            p=p_,
            p_prev=p_prev_,
            mode=self.feature_mode,
            pitch=pitch_,
            omega_r=omega_r_,
            omega_f=omega_f_,
            v_prev=v_prev_,
        )
        pred_t = np.clip(_predict_piecewise_model(self.torque_model, x), 0.0, None)
        pred_f = np.clip(_predict_piecewise_model(self.thrust_model, x), 0.0, None)
        return pred_t, pred_f

    def to_dict(self) -> dict[str, Any]:
        if self.torque_model is None or self.thrust_model is None or self.feature_mode is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "model": "piecewise_linear",
            "feature_mode": self.feature_mode,
            "feature_names": FEATURE_NAMES[self.feature_mode],
            "fit_options": asdict(self.options),
            "target_models": {
                "tshaft": self.torque_model,
                "ft": self.thrust_model,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoadEstimatorPiecewise":
        fit_opts = payload.get("fit_options", {})
        options = PiecewiseFitOptions(
            feature_mode=str(fit_opts.get("feature_mode", payload.get("feature_mode", "vp"))),
            reg_alpha=float(fit_opts.get("reg_alpha", 25.0)),
            normalize_features=bool(fit_opts.get("normalize_features", True)),
            quantiles=tuple(fit_opts.get("quantiles", (0.0, 0.33, 0.66, 1.0))),
        )
        model = cls(options)
        model.feature_mode = str(payload.get("feature_mode", "vp"))
        targets = payload["target_models"]
        model.torque_model = targets["tshaft"]
        model.thrust_model = targets["ft"]
        return model

    def save_json(self, path: str | Path, *, metrics: LoadFitMetrics | dict[str, Any] | None = None) -> Path:
        data: dict[str, Any] = self.to_dict()
        if metrics is not None:
            if isinstance(metrics, dict):
                data["metrics"] = metrics
            else:
                data["metrics"] = asdict(metrics)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> "LoadEstimatorPiecewise":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
