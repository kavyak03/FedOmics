from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROC_DIR = Path("data/processed")
OUT_DIR = PROC_DIR / "qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred_path = PROC_DIR / "predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError("predictions.csv not found. Run training first.")

pred = pd.read_csv(pred_path)

required_cols = {"y_true", "y_prob"}
if not required_cols.issubset(pred.columns):
    raise ValueError("predictions.csv must contain y_true and y_prob columns")

# probability histograms
plt.figure(figsize=(7, 5))
for label in sorted(pred["y_true"].unique()):
    sub = pred[pred["y_true"] == label]
    plt.hist(sub["y_prob"], bins=20, alpha=0.6, label=f"true_label_{label}")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Prediction probability histogram")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "prediction_probability_histogram.png", dpi=150)
plt.close()

# by model if available
if "model" in pred.columns:
    for model_name, sub in pred.groupby("model"):
        plt.figure(figsize=(7, 5))
        for label in sorted(sub["y_true"].unique()):
            ss = sub[sub["y_true"] == label]
            plt.hist(ss["y_prob"], bins=20, alpha=0.6, label=f"true_label_{label}")
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.title(f"Prediction histogram: {model_name}")
        plt.legend()
        plt.tight_layout()
        out_name = "prediction_histogram_{0}.png".format(model_name)
        plt.savefig(OUT_DIR / out_name, dpi=150)
        plt.close()

print("Saved prediction QC plots to", OUT_DIR)