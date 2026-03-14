from pathlib import Path
import sys
import json
import copy
import subprocess

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_yaml

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ABLATION_DIR = PROCESSED_DIR / "ablation"

# ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def detect_federated_key(metrics_dict):
    for k in metrics_dict.keys():
        if k.startswith("federated_"):
            return k
    return None


def summarize_values(vals):
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
    }


def main():
    cfg = load_yaml(str(CONFIG_PATH))

    seeds = cfg.get("ablation_seeds", [42, 123, 999])
    modes = cfg.get("ablation_modes", ["linear", "interaction", "mixed"])

    original_cfg = read_yaml(CONFIG_PATH)
    all_records = []

    python_exe = sys.executable

    try:
        for mode in modes:
            for seed in seeds:
                run_cfg = copy.deepcopy(original_cfg)

                # Ablation-specific overrides
                run_cfg["random_seed"] = int(seed)
                run_cfg["sim_ablation_mode"] = True

                # keep your current ablation-friendly settings if present
                run_cfg["samples_per_center"] = int(run_cfg.get("samples_per_center", 200))
                run_cfg["chi2_top_k"] = int(run_cfg.get("chi2_top_k", 100))
                run_cfg["epochs"] = int(run_cfg.get("epochs", 40))
                run_cfg["learning_rate"] = float(run_cfg.get("learning_rate", 0.0007))

                write_yaml(CONFIG_PATH, run_cfg)

                cmd = [
                    python_exe,
                    "-m",
                    "scripts.run_pipeline",
                    "--clean",
                    "--mode",
                    "sim",
                    "--sim-signal-mode",
                    mode,
                ]

                print("\n=== Running ablation:", "mode={0}".format(mode), "seed={0}".format(seed), "===\n")

                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                )

                log_path = ABLATION_DIR / f"ablation_{mode}_seed_{seed}.log"

                log_path.parent.mkdir(parents=True, exist_ok=True)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                    f.write("\n\n--- STDERR ---\n\n")
                    f.write(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(
                        "Ablation run failed for mode={0}, seed={1}. "
                        "See log: {2}".format(mode, seed, log_path)
                    )

                metrics_path = PROCESSED_DIR / "metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError("metrics.json not found after ablation run.")

                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                federated_key = detect_federated_key(metrics)
                if federated_key is None:
                    raise ValueError("Could not find federated model key in metrics.json")

                federated_block = metrics.get(federated_key, {})
                logreg_block = metrics.get("centralized_logreg", {})

                for center_name, center_metrics in federated_block.items():
                    all_records.append(
                        {
                            "signal_mode": mode,
                            "seed": seed,
                            "model": federated_key,
                            "center": center_name,
                            "accuracy": center_metrics.get("accuracy"),
                            "f1": center_metrics.get("f1"),
                            "auroc": center_metrics.get("auroc"),
                        }
                    )

                for center_name, center_metrics in logreg_block.items():
                    all_records.append(
                        {
                            "signal_mode": mode,
                            "seed": seed,
                            "model": "centralized_logreg",
                            "center": center_name,
                            "accuracy": center_metrics.get("accuracy"),
                            "f1": center_metrics.get("f1"),
                            "auroc": center_metrics.get("auroc"),
                        }
                    )

        raw_df = pd.DataFrame(all_records)
        raw_csv = ABLATION_DIR / "interaction_ablation_raw_results.csv"
        raw_df.to_csv(raw_csv, index=False)

        # Summary by mode + model across all centers and seeds
        summary_rows = []
        for (signal_mode, model), group in raw_df.groupby(["signal_mode", "model"]):
            acc_stats = summarize_values(group["accuracy"].tolist())
            f1_stats = summarize_values(group["f1"].tolist())
            auc_stats = summarize_values(group["auroc"].tolist())

            summary_rows.append(
                {
                    "signal_mode": signal_mode,
                    "model": model,
                    "accuracy_mean": acc_stats["mean"],
                    "accuracy_std": acc_stats["std"],
                    "f1_mean": f1_stats["mean"],
                    "f1_std": f1_stats["std"],
                    "auroc_mean": auc_stats["mean"],
                    "auroc_std": auc_stats["std"],
                    "n_rows": int(len(group)),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(["signal_mode", "model"])
        summary_csv = ABLATION_DIR / "interaction_ablation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        # Also create a compact markdown report
        report_md = ABLATION_DIR / "interaction_ablation_summary.md"
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("# Interaction Ablation Summary\n\n")
            f.write("Runs were performed across multiple seeds and centers.\n\n")
            f.write("## Summary Table\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n## Interpretation Guide\n\n")
            f.write("- Linear: logistic regression should be competitive with the neural model.\n")
            f.write("- Interaction: the neural model should improve relative to logistic regression.\n")
            f.write("- Mixed: the neural model should be competitive or modestly better.\n")

        print("\nSaved raw results to:", raw_csv)
        print("Saved summary CSV to:", summary_csv)
        print("Saved markdown summary to:", report_md)

    finally:
        # Restore original config no matter what
        write_yaml(CONFIG_PATH, original_cfg)
        print("\nRestored original config.yaml")
        

if __name__ == "__main__":
    main()