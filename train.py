"""
train.py

Research-grade training utility for multimodal models.
- Identical training protocol across model variants
- Safe JSON history serialization
- Best-model checkpointing
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def _to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert NumPy/TF scalar types to Python-native types
    so json.dump works reliably.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]

    # NumPy scalar -> Python scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def train_model(model, name: str, train_data, val_data):
    """
    Train model with a fixed protocol for fair comparison.

    Args:
        model: compiled tf.keras model
        name: model name used for saved artifacts
        train_data: tuple (Xv_train, Xa_train, Xt_train, y_train)
        val_data: tuple (Xv_val, Xa_val, Xt_val, y_val)

    Returns:
        keras.callbacks.History
    """
    Xv_train, Xa_train, Xt_train, y_train = train_data
    Xv_val, Xa_val, Xt_val, y_val = val_data

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    model_path = f"outputs/models/{name}.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    history = model.fit(
        x=[Xv_train, Xa_train, Xt_train],
        y=y_train,
        validation_data=([Xv_val, Xa_val, Xt_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    print(f"[INFO] Best model saved to: {model_path}")

    hist: Dict[str, List[Any]] = history.history
    hist_serializable = _to_json_serializable(hist)

    # Save JSON safely (fixes float32 serialization error)
    with open(f"outputs/history/{name}.json", "w", encoding="utf-8") as f:
        json.dump(hist_serializable, f, indent=4)

    # Plot training curves
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(hist.get("mae", []), label="train")
    plt.plot(hist.get("val_mae", []), label="val")
    plt.title("MAE")
    plt.legend()

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{name}_training.png")
    plt.close()

    return history