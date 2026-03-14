from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_yaml

CFG = load_yaml("configs/config.yaml")
RNG = np.random.default_rng(CFG.get("random_seed", 42))

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCGA_DIR = Path("data/raw/tcga_prad/centers")
SIM_DIR = Path("data/demo_dataset")
SIM_DIR.mkdir(parents=True, exist_ok=True)


def _simulate_from_class(class_df, feature_cols, n_needed, sigma):
    rows = []
    if len(class_df) == 0 or n_needed <= 0:
        return rows

    for _ in range(n_needed):
        idx = int(RNG.integers(0, len(class_df)))
        base = class_df.iloc[idx][feature_cols].values.astype(float)
        synth = base + RNG.normal(0, sigma, size=len(feature_cols))
        synth = np.maximum(synth, 0.0)

        row = dict(zip(feature_cols, synth))
        row["label"] = int(class_df.iloc[idx]["label"])
        rows.append(row)

    return rows


def augment_training_only(train_df):
    """
    Conservative train-only augmentation for TCGA mode.
    Optional and configurable.
    """
    if not bool(CFG.get("tcga_train_aug_enabled", True)):
        print("TCGA train augmentation disabled in config.")
        return train_df

    feature_cols = [c for c in train_df.columns if c != "label"]
    class_counts = train_df["label"].value_counts().to_dict()

    if len(class_counts) < 2:
        print("Skipping augmentation: training split has only one class.")
        return train_df

    real_n = len(train_df)
    aug_ratio = float(CFG.get("tcga_train_aug_ratio", 0.5))
    synth_target_total = int(np.floor(real_n * aug_ratio))

    if synth_target_total <= 0:
        print("Skipping augmentation: tcga_train_aug_ratio produced zero synthetic samples.")
        return train_df

    sigma = np.std(train_df[feature_cols].values, axis=0, ddof=1)
    sigma = np.nan_to_num(sigma, nan=0.0)
    sigma = sigma * float(CFG.get("hybrid_noise_fraction", 0.1)) + 1e-6

    balance_mode = str(CFG.get("tcga_train_aug_balance_mode", "conservative")).lower()

    synth_rows = []

    if balance_mode == "balanced":
        # allocate more synth to the minority class
        counts = train_df["label"].value_counts().to_dict()
        labels = sorted(train_df["label"].unique())
        if len(labels) != 2:
            return train_df

        label0, label1 = labels[0], labels[1]
        n0 = counts.get(label0, 0)
        n1 = counts.get(label1, 0)

        if n0 > n1:
            extra_minority = int(np.ceil(synth_target_total * 0.7))
            extra_majority = synth_target_total - extra_minority
            alloc = {label0: extra_majority, label1: extra_minority}
        elif n1 > n0:
            extra_minority = int(np.ceil(synth_target_total * 0.7))
            extra_majority = synth_target_total - extra_minority
            alloc = {label0: extra_minority, label1: extra_majority}
        else:
            half = synth_target_total // 2
            alloc = {label0: half, label1: synth_target_total - half}

        for label, needed in alloc.items():
            class_df = train_df[train_df["label"] == label]
            synth_rows.extend(
                _simulate_from_class(class_df, feature_cols, needed, sigma)
            )

    else:
        # conservative: allocate synth proportional to observed class distribution
        counts = train_df["label"].value_counts(normalize=True).to_dict()
        for label in sorted(train_df["label"].unique()):
            prop = counts.get(label, 0.0)
            needed = int(round(synth_target_total * prop))
            class_df = train_df[train_df["label"] == label]
            synth_rows.extend(
                _simulate_from_class(class_df, feature_cols, needed, sigma)
            )

    if not synth_rows:
        return train_df

    synth_df = pd.DataFrame(synth_rows, columns=train_df.columns)
    out = pd.concat([train_df, synth_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=CFG.get("random_seed", 42)).reset_index(drop=True)

    print(
        "Training-only augmentation applied:",
        "real_train =", len(train_df),
        "synthetic_train =", len(synth_df),
        "final_train =", len(out),
        "train_label_counts_before =", class_counts,
        "train_label_counts_after =", out["label"].value_counts().to_dict(),
    )
    return out


source_dir = TCGA_DIR if list(TCGA_DIR.glob("center_*_expression.csv")) else SIM_DIR
is_tcga_mode = source_dir == TCGA_DIR

print("Using input data from:", source_dir)
if is_tcga_mode:
    print("TCGA mode detected: True")
else:
    print("Sim mode detected: True")

input_files = sorted(source_dir.glob("center_*_expression.csv"))
if not input_files:
    raise FileNotFoundError("No center expression files found. Generate or download data first.")

for center_file in input_files:
    df = pd.read_csv(center_file)

    if df.empty:
        raise ValueError("{0} is empty.".format(center_file))
    if "label" not in df.columns:
        raise ValueError("{0} is missing a 'label' column.".format(center_file))

    n_rows = len(df)
    class_counts = df["label"].value_counts().to_dict()
    n_classes = len(class_counts)

    if is_tcga_mode and n_rows < 30:
        test_size = 0.25
    else:
        test_size = 0.2 if n_rows >= 20 else 0.5

    test_count = max(1, int(round(n_rows * test_size)))

    use_stratify = (
        n_classes > 1
        and all(v >= 2 for v in class_counts.values())
        and test_count >= n_classes
    )

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"] if use_stratify else None,
    )

    if is_tcga_mode:
        train_df = augment_training_only(train_df)

    train_df.to_csv(OUT_DIR / "{0}_train.csv".format(center_file.stem), index=False)
    val_df.to_csv(OUT_DIR / "{0}_val.csv".format(center_file.stem), index=False)

    print(
        "{0}: total={1}, train={2}, val={3}, labels={4}, stratified={5}".format(
            center_file.name,
            n_rows,
            len(train_df),
            len(val_df),
            class_counts,
            use_stratify,
        )
    )
    print(
        "Validation label counts for {0}: {1}".format(
            center_file.name,
            val_df["label"].value_counts().to_dict()
        )
    )

print("Preprocessing complete")