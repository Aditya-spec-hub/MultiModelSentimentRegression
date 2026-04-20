"""
temporal.py
"""

from __future__ import annotations

from typing import Dict, Union, Sequence
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ArrayLike = Union[np.ndarray, list, tuple]


def _to_1d_array(predictions: ArrayLike) -> np.ndarray:
    arr = np.asarray(predictions, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(f"Predictions must be 1D or (N,1), got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def apply_ema(predictions: ArrayLike, alpha: float = 0.6) -> np.ndarray:
    x = _to_1d_array(predictions)
    if x.size < 2:
        return x.copy()

    alpha = float(np.clip(alpha, 0.0, 1.0))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def adaptive_smoothing(predictions: ArrayLike, base_alpha: float = 0.6) -> np.ndarray:
    x = _to_1d_array(predictions)
    if x.size < 2:
        return x.copy()

    base_alpha = float(np.clip(base_alpha, 0.0, 1.0))
    alpha_min = max(0.05, base_alpha - 0.3)
    alpha_max = min(0.95, base_alpha + 0.3)

    diff = np.abs(np.diff(x))
    scale = max(float(np.median(diff)), 1e-8)
    norm = diff / scale
    s = norm / (1.0 + norm)
    alpha_t = alpha_min + (alpha_max - alpha_min) * s

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        a = float(alpha_t[i - 1])
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def _group_key_from_id(uid: str) -> str:
    """
    MOSEI keys are often like: 'videoId[segment]' -> group by videoId.
    Fallback: take prefix before last '_' if present.
    """
    s = str(uid)
    m = re.match(r"^(.*?)\[[^\]]+\]$", s)
    if m:
        return m.group(1)
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s


def smooth_by_group(predictions: ArrayLike, ids: Sequence[str], alpha: float = 0.6) -> np.ndarray:
    x = _to_1d_array(predictions)
    if len(x) != len(ids):
        raise ValueError("Length mismatch: predictions and ids")

    out = np.empty_like(x)
    ids = np.asarray(ids, dtype=object)

    groups = {}
    for i, uid in enumerate(ids):
        g = _group_key_from_id(str(uid))
        groups.setdefault(g, []).append(i)

    for _, idxs in groups.items():
        seq = x[idxs]
        seq_sm = adaptive_smoothing(apply_ema(seq, alpha=alpha), base_alpha=alpha)
        out[idxs] = seq_sm

    return out


def temporal_consistency_loss(y_pred: tf.Tensor, mode: str = "l2", delta: float = 0.1) -> tf.Tensor:
    rank = y_pred.shape.rank
    if rank is not None and rank < 2:
        return tf.constant(0.0, dtype=tf.float32)
    if rank == 2 and y_pred.shape[-1] == 1:
        return tf.constant(0.0, dtype=tf.float32)

    yp = y_pred[:, :, 0] if rank == 3 else y_pred
    d = yp[:, 1:] - yp[:, :-1]

    if mode == "l1":
        return tf.reduce_mean(tf.abs(d))
    if mode == "huber":
        return tf.reduce_mean(tf.keras.losses.huber(tf.zeros_like(d), d, delta=delta))
    return tf.reduce_mean(tf.square(d))


def make_temporal_regularized_loss(
    base_loss: str = "mse",
    lambda_tc: float = 0.05,
    tc_mode: str = "l2",
    huber_delta: float = 0.1,
):
    base_fn = tf.keras.losses.MeanSquaredError() if base_loss == "mse" else tf.keras.losses.MeanAbsoluteError()
    lambda_tc = float(max(0.0, lambda_tc))

    def _loss(y_true, y_pred):
        return base_fn(y_true, y_pred) + lambda_tc * temporal_consistency_loss(
            y_pred, mode=tc_mode, delta=huber_delta
        )

    return _loss


def compute_stability(predictions: ArrayLike) -> float:
    x = _to_1d_array(predictions)
    if x.size < 2:
        return 0.0
    return float(np.var(np.diff(x)))


def compute_switching_frequency(predictions: ArrayLike, threshold: float = 0.5) -> int:
    x = _to_1d_array(predictions)
    if x.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(x)) > float(threshold)))


def temporal_pipeline(predictions: ArrayLike, ids=None, name: str = "temporal") -> Dict[str, object]:
    raw = _to_1d_array(predictions)
    if raw.size == 0:
        return {
            "smoothed": raw.copy(),
            "raw_stability": 0.0,
            "stability": 0.0,
            "raw_switches": 0,
            "switches": 0,
            "improvement": 0.0,
            "relative_improvement": 0.0,
        }

    if ids is None:
        sm = adaptive_smoothing(apply_ema(raw, alpha=0.6), base_alpha=0.6)
    else:
        sm = smooth_by_group(raw, ids=ids, alpha=0.6)

    raw_stability = compute_stability(raw)
    sm_stability = compute_stability(sm)
    improvement = float(raw_stability - sm_stability)
    rel = float(improvement / raw_stability) if raw_stability > 1e-8 else 0.0

    os.makedirs("outputs/plots", exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(raw, label="Raw", alpha=0.6)
    plt.plot(sm, label="Smoothed", linewidth=2.0)
    plt.title("Temporal Smoothing Effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{name}.png", dpi=180)
    plt.close()

    return {
        "smoothed": sm,
        "raw_stability": raw_stability,
        "stability": sm_stability,
        "raw_switches": compute_switching_frequency(raw),
        "switches": compute_switching_frequency(sm),
        "improvement": improvement,
        "relative_improvement": rel,
    }