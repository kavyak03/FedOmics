
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROC = Path("data/processed")
OUT = Path("outputs/qc")
OUT.mkdir(parents=True, exist_ok=True)

train_files = sorted(PROC.glob("center_*_train.csv"))
val_files = sorted(PROC.glob("center_*_val.csv"))
if not train_files:
    raise FileNotFoundError("No processed train files found. Run preprocess_data.py first.")

center_counts = []
label_counts = {}
for f in train_files + val_files:
    df = pd.read_csv(f)
    center_name = f.stem.replace("_train", "").replace("_val", "")
    center_counts.append({"center": center_name, "split": "train" if "_train" in f.stem else "val", "n": len(df)})
    vc = df["label"].value_counts().to_dict()
    label_counts[f.stem] = vc

center_df = pd.DataFrame(center_counts)
summary = {
    "center_counts": center_df.to_dict(orient="records"),
    "label_counts": label_counts,
}

with open(OUT / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pivot = center_df.pivot(index="center", columns="split", values="n").fillna(0)
pivot.plot(kind="bar", figsize=(7,4))
plt.title("Samples per center")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(OUT / "samples_per_center.png", dpi=150)
plt.close()

all_labels = []
for f in train_files + val_files:
    df = pd.read_csv(f)
    split = "train" if "_train" in f.stem else "val"
    tmp = df["label"].value_counts().rename_axis("label").reset_index(name="count")
    tmp["dataset"] = f.stem.replace("_train", "").replace("_val", "")
    tmp["split"] = split
    all_labels.append(tmp)
lab = pd.concat(all_labels, ignore_index=True)
lab.groupby("label")["count"].sum().plot(kind="bar", figsize=(5,4))
plt.title("Overall label balance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "label_balance.png", dpi=150)
plt.close()

print("QC outputs written to outputs/qc/")
