from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from data_loader import load_dataset
from preprocessing import extract_features, pad_and_prepare, train_val_test_split, normalize_data
from model import build_model
from train import train_model
from evaluate import evaluate_model
from temporal import temporal_pipeline


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=4)


def _safe_1d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(-1)
    return a.reshape(-1)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    y_true = _safe_1d(y_true).astype(np.float64)
    y_pred = _safe_1d(y_pred).astype(np.float64)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if (np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8) else 0.0
    return mae, rmse, pearson


def _classification_metrics_from_regression(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = _safe_1d(y_true)
    y_pred = _safe_1d(y_pred)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    yt = (y_true >= 0).astype(int)
    yp = (y_pred >= 0).astype(int)

    acc2 = float(np.mean(yt == yp))
    f1 = float(f1_score(yt, yp, average="binary"))
    return acc2, f1


def _full_metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae, rmse, pearson = _regression_metrics(y_true, y_pred)
    acc2, f1 = _classification_metrics_from_regression(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": pearson,
        "Acc-2": acc2,
        "F1": f1,
        "Pred_Mean": float(np.mean(y_pred)),
        "Pred_Std": float(np.std(y_pred)),
        "Pred_Var": float(np.var(y_pred)),
    }


def print_results(title: str, results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n================ {title} ================")
    print("Model    |      MAE |     RMSE |  Pearson |   Acc-2 |      F1 |  Pred_Std")
    print("----------------------------------------------------------------------------")
    for k in ["lstm", "gru", "bigru"]:
        m = results[k]
        print(
            f"{k.upper():8s} | {m['MAE']:8.4f} | {m['RMSE']:8.4f} | {m['Pearson']:8.4f} | "
            f"{m['Acc-2']:7.4f} | {m['F1']:7.4f} | {m['Pred_Std']:9.6f}"
        )
    print("============================================================================")


def run_experiment(data_path: str, seed: int = 42, use_attention: bool = True, temporal_safe_mode: bool = True):
    set_global_seed(seed)

    data_path = str(Path(data_path).expanduser().resolve())
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    dataset = load_dataset(data_path)

    # include IDs
    Xv_list, Xa_list, Xt_list, y_list, ids = extract_features(dataset)
    Xv, Xa, Xt, y = pad_and_prepare(Xv_list, Xa_list, Xt_list, y_list)

    # IMPORTANT:
    # - shuffle=True for standard IID training
    # - if temporal_safe_mode=True, use shuffle=False so test order remains meaningful
    split = train_val_test_split(
        Xv, Xa, Xt, y,
        ids=ids,
        seed=seed,
        shuffle=not temporal_safe_mode
    )

    (
        Xv_train, Xv_val, Xv_test,
        Xa_train, Xa_val, Xa_test,
        Xt_train, Xt_val, Xt_test,
        y_train, y_val, y_test,
        ids_train, ids_val, ids_test
    ) = split

    Xv_train, Xv_val, Xv_test = normalize_data(Xv_train, Xv_val, Xv_test)
    Xa_train, Xa_val, Xa_test = normalize_data(Xa_train, Xa_val, Xa_test)
    Xt_train, Xt_val, Xt_test = normalize_data(Xt_train, Xt_val, Xt_test)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    print("[DEBUG] Train shapes:", Xv_train.shape, Xa_train.shape, Xt_train.shape, y_train.shape)
    print(f"[INFO] temporal_safe_mode={temporal_safe_mode}")

    train_data = (Xv_train, Xa_train, Xt_train, y_train)
    val_data = (Xv_val, Xa_val, Xt_val, y_val)

    rnn_types = ["lstm", "gru", "bigru"]
    baseline_results, temporal_results, temporal_details = {}, {}, {}

    for rnn_type in rnn_types:
        print(f"\n[INFO] Training {rnn_type.upper()} (use_attention={use_attention})")

        model = build_model(
            rnn_type=rnn_type,
            use_attention=use_attention,
            max_len=Xv_train.shape[1],
            visual_dim=Xv_train.shape[2],
            audio_dim=Xa_train.shape[2],
            text_dim=Xt_train.shape[2],
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

        train_model(model=model, name=f"{rnn_type}_baseline", train_data=train_data, val_data=val_data)

        baseline_pred = model.predict([Xv_test, Xa_test, Xt_test], verbose=0).reshape(-1)
        baseline_pack = _full_metric_pack(y_test, baseline_pred)
        baseline_results[rnn_type] = baseline_pack

        evaluate_model(
            model=model,
            Xv=Xv_test, Xa=Xa_test, Xt=Xt_test, y=y_test,
            name=f"{rnn_type}_baseline",
        )

        tp = temporal_pipeline(
            baseline_pred,
            ids=ids_test,  # critical fix
            name=f"{rnn_type}_temporal"
        )
        smoothed_pred = np.asarray(tp["smoothed"], dtype=np.float32).reshape(-1)

        temporal_pack = _full_metric_pack(y_test, smoothed_pred)
        temporal_results[rnn_type] = temporal_pack

        temporal_details[rnn_type] = {
            "raw_stability": float(tp["raw_stability"]),
            "smoothed_stability": float(tp["stability"]),
            "raw_switches": int(tp["raw_switches"]),
            "smoothed_switches": int(tp["switches"]),
            "stability_improvement": float(tp["improvement"]),
            "relative_improvement": float(tp["relative_improvement"]),
        }

        save_json(f"outputs/results/{rnn_type}_baseline.json", baseline_pack)
        save_json(f"outputs/results/{rnn_type}_temporal.json", {**temporal_pack, **temporal_details[rnn_type]})

    print_results("BASELINE RESULTS (TEST)", baseline_results)
    print_results("TEMPORAL RESULTS (TEST, POST-HOC)", temporal_results)

    save_json("outputs/results/comparison_baseline.json", {
        "seed": seed,
        "use_attention": use_attention,
        "temporal_safe_mode": temporal_safe_mode,
        "results": baseline_results,
    })
    save_json("outputs/results/comparison_temporal.json", {
        "seed": seed,
        "use_attention": use_attention,
        "temporal_safe_mode": temporal_safe_mode,
        "results": temporal_results,
        "temporal_metrics": temporal_details,
    })

    print("[INFO] Saved all result files in outputs/results/")


def parse_args():
    parser = argparse.ArgumentParser(description="CMU-MOSEI baseline + temporal run")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CMU-MOSEI data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--temporal_safe_mode", action="store_true",
                        help="Use order-preserving split for valid temporal post-hoc analysis.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_path=args.data_path,
        seed=args.seed,
        use_attention=not args.no_attention,
        temporal_safe_mode=args.temporal_safe_mode,
    )