"""
preprocessing.py

Research-grade preprocessing for CMU-MOSEI
"""

from __future__ import annotations

import logging
import numpy as np


LOGGER_NAME = "mosei.preprocessing"
MAX_LEN = 100

VISUAL = "OpenFace_2"
AUDIO = "COVAREP"
TEXT = "glove_vectors"
LABEL = "All Labels"


# ==============================
# LOGGER
# ==============================
def _logger():
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ==============================
# EXTRACT
# ==============================
def extract_features(dataset):
    """
    Returns:
      Xv, Xa, Xt, y, ids
    where ids preserves deterministic utterance order.
    """
    log = _logger()

    Xv, Xa, Xt, y, kept_ids = [], [], [], [], []
    skipped = 0
    lengths = []

    # deterministic aligned ids
    ids = sorted(
        set(dataset[VISUAL].keys())
        & set(dataset[AUDIO].keys())
        & set(dataset[TEXT].keys())
        & set(dataset[LABEL].keys())
    )

    log.info(f"Total aligned samples: {len(ids)}")

    for vid in ids:
        try:
            v = dataset[VISUAL][vid]["features"]
            a = dataset[AUDIO][vid]["features"]
            t = dataset[TEXT][vid]["features"]
            l = dataset[LABEL][vid]["features"]

            if len(v) == 0 or len(a) == 0 or len(t) == 0:
                skipped += 1
                continue

            if v.ndim != 2 or a.ndim != 2 or t.ndim != 2:
                skipped += 1
                continue

            label = float(np.mean(l))
            if not np.isfinite(label):
                skipped += 1
                continue

            Xv.append(v.astype(np.float32))
            Xa.append(a.astype(np.float32))
            Xt.append(t.astype(np.float32))
            y.append(label)
            kept_ids.append(vid)

            lengths.append(len(v))

        except Exception:
            skipped += 1

    if len(y) == 0:
        raise ValueError("No valid samples extracted!")

    log.info(f"Valid samples: {len(Xv)}")
    log.info(f"Skipped samples: {skipped}")
    log.info(f"Label range: {min(y):.3f} to {max(y):.3f}")
    log.info(f"Label std: {np.std(y):.3f}")
    log.info(
        f"Sequence length → mean: {np.mean(lengths):.1f}, "
        f"max: {np.max(lengths)}, min: {np.min(lengths)}"
    )

    return Xv, Xa, Xt, y, kept_ids


# ==============================
# PADDING
# ==============================
def _pad(seq_list, dim):
    N = len(seq_list)
    X = np.zeros((N, MAX_LEN, dim), dtype=np.float32)

    for i, seq in enumerate(seq_list):
        length = min(len(seq), MAX_LEN)
        X[i, :length] = seq[:length]

    return X


def pad_and_prepare(Xv, Xa, Xt, y):
    print("[INFO] Padding...")

    # infer dims from data (safer than hardcoding 713)
    v_dim = int(Xv[0].shape[1])
    a_dim = int(Xa[0].shape[1])
    t_dim = int(Xt[0].shape[1])

    Xv = _pad(Xv, v_dim)
    Xa = _pad(Xa, a_dim)
    Xt = _pad(Xt, t_dim)
    y = np.array(y, dtype=np.float32)

    return Xv, Xa, Xt, y


# ==============================
# SPLIT (NO LEAKAGE)
# ==============================
def train_val_test_split(Xv, Xa, Xt, y, ids=None, seed=42, shuffle=True):
    """
    If shuffle=False, preserves temporal/global order (useful for post-hoc temporal analysis).
    """
    n = len(y)
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

    Xv, Xa, Xt, y = Xv[idx], Xa[idx], Xt[idx], y[idx]
    ids_arr = np.asarray(ids, dtype=object)[idx] if ids is not None else None

    t1 = int(0.7 * n)
    t2 = int(0.85 * n)

    out = (
        Xv[:t1], Xv[t1:t2], Xv[t2:],
        Xa[:t1], Xa[t1:t2], Xa[t2:],
        Xt[:t1], Xt[t1:t2], Xt[t2:],
        y[:t1], y[t1:t2], y[t2:]
    )

    if ids_arr is None:
        return out

    return out + (ids_arr[:t1], ids_arr[t1:t2], ids_arr[t2:])


# ==============================
# NORMALIZATION (SAFE)
# ==============================
def normalize_data(X_train, X_val, X_test):
    if X_train.size == 0:
        raise ValueError("Training data empty")

    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    return X_train, X_val, X_test