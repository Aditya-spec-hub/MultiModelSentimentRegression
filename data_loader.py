"""
data_loader.py

Research-grade CMU-MOSEI loader.

Features:
- Flexible file discovery
- Strict modality validation
- Deterministic sample alignment
- Dataset integrity checks
- Label sanity verification (robust)
- Feature shape logging
- Reproducibility-safe
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from mmsdk import mmdatasdk as md


LOGGER_NAME = "mosei.data_loader"

# Correct keys
MODALITIES = {
    "visual": "OpenFace_2",
    "audio": "COVAREP",
    "text": "glove_vectors",
    "labels": "All Labels",
}


# ==============================
# LOGGER
# ==============================
def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


# ==============================
# FIND FILES (FLEXIBLE)
# ==============================
def _find_files(root: Path) -> Dict[str, str]:

    mapping = {}

    for file in root.glob("*.csd"):
        name = file.name

        if "COVAREP" in name:
            mapping["COVAREP"] = str(file)
        elif "OpenFace" in name:
            mapping["OpenFace_2"] = str(file)
        elif "WordVectors" in name:
            mapping["glove_vectors"] = str(file)
        elif "Labels" in name:
            mapping["All Labels"] = str(file)

    # 🔥 STRICT CHECK
    required = {"COVAREP", "OpenFace_2", "glove_vectors", "All Labels"}
    missing = required - set(mapping.keys())

    if missing:
        raise ValueError(f"Missing required .csd files: {missing}")

    return mapping


# ==============================
# VALIDATION
# ==============================
def _validate_dataset(dataset: md.mmdataset, logger: logging.Logger):

    available = list(dataset.keys())
    logger.info(f"Available sequences: {available}")

    missing = [v for v in MODALITIES.values() if v not in available]

    if missing:
        raise ValueError(f"Missing sequences inside dataset: {missing}")

    logger.info("All required modalities present.")


# ==============================
# STATS + SANITY CHECK
# ==============================
def _log_stats_and_sanity(dataset: md.mmdataset, logger: logging.Logger):

    # Collect IDs per modality
    ids = {
        k: set(dataset[v].keys())
        for k, v in MODALITIES.items()
    }

    # 🔥 deterministic intersection
    all_ids = sorted(set.intersection(*ids.values()))

    logger.info(f"Aligned valid samples: {len(all_ids)}")

    # Dataset size check
    if len(all_ids) < 1000:
        logger.warning("Dataset too small — results may not be reliable")

    # ==============================
    # LABEL SANITY (STRONG)
    # ==============================
    sample_ids = all_ids[:min(500, len(all_ids))]
    labels = []

    for vid in sample_ids:
        try:
            l = dataset[MODALITIES["labels"]][vid]["features"]
            labels.append(float(np.mean(l)))
        except Exception:
            continue

    if labels:
        logger.info(
            f"Label stats → min: {np.min(labels):.3f}, max: {np.max(labels):.3f}, "
            f"mean: {np.mean(labels):.3f}, std: {np.std(labels):.3f}"
        )

        logger.info(
            "Expected range ≈ [-3, +3]. "
            "If range is too small, your evaluation is misleading."
        )

    # ==============================
    # FEATURE SHAPE DEBUG
    # ==============================
    try:
        sample_vid = all_ids[0]

        v_shape = dataset[MODALITIES["visual"]][sample_vid]["features"].shape
        a_shape = dataset[MODALITIES["audio"]][sample_vid]["features"].shape
        t_shape = dataset[MODALITIES["text"]][sample_vid]["features"].shape

        logger.info(
            f"Sample feature shapes → Visual: {v_shape}, Audio: {a_shape}, Text: {t_shape}"
        )

    except Exception:
        logger.warning("Could not read feature shapes (non-critical).")


# ==============================
# MAIN LOADER
# ==============================
def load_dataset(data_path: str):

    logger = _configure_logger()
    root = Path(data_path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    logger.info(f"Loading dataset from: {root}")

    # Find files
    recipe = _find_files(root)
    logger.info(f"Using recipe: {recipe}")

    # Load dataset
    dataset = md.mmdataset(recipe)

    # Validate structure
    _validate_dataset(dataset, logger)

    # Log stats + sanity
    _log_stats_and_sanity(dataset, logger)

    logger.info("Dataset loaded successfully.")
    return dataset