"""
utils.py (FINAL VERSION)

- Safe GPU setup
- Mixed precision (with stability)
- Reproducibility support
- Smart batch scaling
"""

from __future__ import annotations
import os

# Suppress TF/CUDA noise before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide INFO + WARNING C++ logs
os.environ.setdefault("PYTHONIOENCODING", "utf-8")    # safe Unicode output

import tensorflow as tf


# ==============================
# GPU CONFIG
# ==============================
def configure_gpu(
    allow_memory_growth=True,
    use_mixed_precision=True,
    seed: int = 42,
):

    # ------------------------------
    # Reproducibility (IMPORTANT)
    # ------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

    # ------------------------------
    # GPU detection
    # ------------------------------
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[GPU] No GPU -> using CPU")
        return "cpu"

    # ------------------------------
    # Memory growth
    # ------------------------------
    if allow_memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    # ------------------------------
    # Mixed precision (SAFE)
    # ------------------------------
    if use_mixed_precision:
        from tensorflow.keras import mixed_precision

        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        print("[GPU] Mixed precision enabled (safe mode)")

    # ------------------------------
    # Info
    # ------------------------------
    for i, gpu in enumerate(gpus):
        print(f"[GPU] Device {i}: {gpu.name}")

    return "gpu"


# ==============================
# BATCH SIZE
# ==============================
def recommend_batch_size(base=32, device="cpu"):

    if device == "gpu":
        # smarter scaling
        size = base * 2

        if size > 128:
            size = 128  # prevent instability

        print(f"[GPU] Using batch size: {size}")
        return size

    print(f"[CPU] Using batch size: {base}")
    return base


# ==============================
# CPU OPTIMIZATION
# ==============================
def optimize_cpu():

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    print("[CPU] Threading optimized")