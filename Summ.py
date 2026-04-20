from __future__ import annotations
import json
from pathlib import Path

RESULTS_DIR = Path("outputs/results")
BASELINE_FILE = RESULTS_DIR / "comparison_baseline.json"
TEMPORAL_FILE = RESULTS_DIR / "comparison_temporal.json"

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_metric(d: dict, keys, default=0.0):
    for k in keys:
        if k in d:
            return d[k]
    return default

def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "NA"

def model_row(model: str, b: dict, t: dict):
    return [
        model.upper(),
        fmt(get_metric(b, ["MAE", "mae"])), fmt(get_metric(t, ["MAE", "mae"])),
        fmt(get_metric(b, ["RMSE", "rmse"])), fmt(get_metric(t, ["RMSE", "rmse"])),
        fmt(get_metric(b, ["Pearson", "pearson"])), fmt(get_metric(t, ["Pearson", "pearson"])),
        fmt(get_metric(b, ["Acc-2", "acc2"])), fmt(get_metric(t, ["Acc-2", "acc2"])),
        fmt(get_metric(b, ["F1", "f1"])), fmt(get_metric(t, ["F1", "f1"])),
        fmt(get_metric(b, ["Pred_Std", "pred_std"]), 6), fmt(get_metric(t, ["Pred_Std", "pred_std"]), 6),
    ]

def best_baseline(results: dict):
    best_m, best_v = None, 1e18
    for m, d in results.items():
        mae = float(get_metric(d, ["MAE", "mae"], 1e18))
        if mae < best_v:
            best_v, best_m = mae, m
    return best_m, best_v

def main():
    baseline = load_json(BASELINE_FILE)
    temporal = load_json(TEMPORAL_FILE)

    bres = baseline["results"]
    tres = temporal["results"]
    models = ["lstm", "gru", "bigru"]

    header = [
        "Model", "MAE(B)", "MAE(T)", "RMSE(B)", "RMSE(T)", "Pearson(B)", "Pearson(T)",
        "Acc2(B)", "Acc2(T)", "F1(B)", "F1(T)", "PredStd(B)", "PredStd(T)",
    ]

    rows = [model_row(m, bres[m], tres[m]) for m in models]

    md_lines = ["# Results Summary", ""]
    md_lines.append(f"- temporal_safe_mode: **{temporal.get('temporal_safe_mode', False)}**")
    md_lines.append("- NOTE: Temporal results are post-hoc smoothed predictions.")
    md_lines.append("")
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")

    best_m, best_mae = best_baseline(bres)
    md_lines.append("")
    md_lines.append(f"**Best baseline model (by MAE): {best_m.upper()} ({best_mae:.4f})**")

    if "temporal_metrics" in temporal:
        md_lines.append("")
        md_lines.append("## Temporal Stability Metrics")
        md_lines.append("")
        md_lines.append("| Model | Raw Stability | Smoothed Stability | Relative Improvement | Raw Switches | Smoothed Switches |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for m in models:
            tm = temporal["temporal_metrics"].get(m, {})
            md_lines.append(
                f"| {m.upper()} | {float(tm.get('raw_stability', 0.0)):.6f} | "
                f"{float(tm.get('smoothed_stability', 0.0)):.6f} | "
                f"{float(tm.get('relative_improvement', 0.0)):.4f} | "
                f"{int(tm.get('raw_switches', 0))} | {int(tm.get('smoothed_switches', 0))} |"
            )

    out_md = RESULTS_DIR / "summary.md"
    out_txt = RESULTS_DIR / "summary.txt"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_txt.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Saved markdown summary: {out_md}")
    print(f"[OK] Saved text summary:     {out_txt}")

if __name__ == "__main__":
    main()