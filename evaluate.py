import numpy as np
import os
import json
import matplotlib.pyplot as plt


# ==============================
# SAFE PEARSON
# ==============================
def _safe_pearson(y_true, y_pred):
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ==============================
# BINARY METRICS
# ==============================
def _binary_metrics(y_true, y_pred):
    y_true_bin = (y_true >= 0).astype(int)
    y_pred_bin = (y_pred >= 0).astype(int)

    acc = np.mean(y_true_bin == y_pred_bin)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(acc), float(f1)


# ==============================
# PLOTS (FOR PAPER)
# ==============================
def _plot_regression(y_true, y_pred, name):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Regression: {name}")

    plt.savefig(f"outputs/plots/{name}_scatter.png")
    plt.close()


def _plot_error_distribution(errors, name):
    plt.figure()
    plt.hist(errors, bins=50)
    plt.title(f"Error Distribution: {name}")
    plt.xlabel("Error")

    plt.savefig(f"outputs/plots/{name}_error.png")
    plt.close()


# ==============================
# MAIN EVALUATION
# ==============================
def evaluate_model(model, Xv, Xa, Xt, y, name="model"):

    os.makedirs("outputs/results", exist_ok=True)

    y_true = np.asarray(y).flatten()
    y_pred = model.predict([Xv, Xa, Xt], verbose=0).flatten()

    # 🔥 SAFETY CHECKS
    if len(y_true) == 0:
        raise ValueError("Empty ground truth")

    if y_true.shape != y_pred.shape:
        raise ValueError("Mismatch between y_true and y_pred")

    # Remove NaNs if any
    y_true = np.nan_to_num(y_true)
    y_pred = np.nan_to_num(y_pred)

    # ==============================
    # METRICS
    # ==============================
    err = y_true - y_pred

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    pearson = _safe_pearson(y_true, y_pred)

    acc2, f1 = _binary_metrics(y_true, y_pred)

    # ==============================
    # EXTRA ANALYSIS
    # ==============================
    pred_mean = float(np.mean(y_pred))
    pred_std = float(np.std(y_pred))

    # ==============================
    # SAVE PLOTS
    # ==============================
    _plot_regression(y_true, y_pred, name)
    _plot_error_distribution(err, name)

    # ==============================
    # RESULTS
    # ==============================
    results = {
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": pearson,
        "Acc-2": acc2,
        "F1": f1,
        "Pred_Mean": pred_mean,
        "Pred_Std": pred_std,
    }

    # SAVE JSON
    with open(f"outputs/results/{name}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results