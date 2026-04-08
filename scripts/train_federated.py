import argparse
from pathlib import Path
import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import skops.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.federated import federated_average
from src.mlops import log_standalone_training_run
from src.model import GeneExpressionNet
from src.utils import (
    choose_best_threshold,
    compute_binary_metrics,
    load_yaml,
    resolve_config_path,
    standardize_train_val,
)

PROC_DIR = Path("data/processed")
MODELS_DIR = PROC_DIR / "models"
RAW_TCGA_CENTERS_DIR = Path("data/raw/tcga_prad/centers")

CFG_PATH = resolve_config_path("configs/config.yaml")
CFG = load_yaml(str(CFG_PATH))

train_files = sorted(PROC_DIR.glob("center_*_train.csv"))
val_files = sorted(PROC_DIR.glob("center_*_val.csv"))

if not train_files or not val_files:
    raise FileNotFoundError("Train/validation files not found. Run preprocess_data.py first.")

parser = argparse.ArgumentParser()
parser.add_argument("--model-backend", choices=["pytorch"], default="pytorch")
ARGS = parser.parse_args()


def get_global_selected_genes():
    selected_path = PROC_DIR / "selected_genes.csv"
    if not selected_path.exists():
        raise FileNotFoundError("selected_genes.csv not found. Run feature_selection.py first.")
    selected_df = pd.read_csv(selected_path)
    return selected_df["selected_gene"].tolist()


def select_top_k_features_train_only(train_df, top_k):
    if "label" not in train_df.columns:
        raise ValueError("train_df must contain a 'label' column")

    metadata_cols = {"label", "center", "signal_mode", "model_backend", "dataset_mode"}
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols and c != "label"]

    if not feature_cols:
        raise ValueError("No numeric feature columns found in training dataframe.")

    X = train_df[feature_cols]
    y = train_df["label"].values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    scores, _ = chi2(X_scaled, y)
    score_df = (
        pd.DataFrame({"gene": X.columns.tolist(), "chi2_score": scores})
        .sort_values("chi2_score", ascending=False)
        .reset_index(drop=True)
    )

    k = min(int(top_k), len(score_df))
    selected = score_df.head(k)["gene"].tolist()
    return selected


def fit_mlp_model(X_train, y_train, input_dim):
    model = GeneExpressionNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(CFG.get("learning_rate", 0.001)),
    )
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=min(16, len(X_train)),
        shuffle=True,
    )

    model.train()
    for _ in range(int(CFG.get("epochs", 20))):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def summarize_metric_dicts(metric_list):
    keys = ["accuracy", "f1", "auroc"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if m.get(k) is not None]
        if vals:
            out[k + "_mean"] = float(np.mean(vals))
            out[k + "_std"] = float(np.std(vals, ddof=0))
        else:
            out[k + "_mean"] = None
            out[k + "_std"] = None
    return out


def compute_average_metrics(metrics_dict):
    acc = []
    f1 = []
    auc = []

    for center_metrics in metrics_dict.values():
        if center_metrics.get("accuracy") is not None:
            acc.append(center_metrics["accuracy"])
        if center_metrics.get("f1") is not None:
            f1.append(center_metrics["f1"])
        if center_metrics.get("auroc") is not None:
            auc.append(center_metrics["auroc"])

    return {
        "accuracy_mean": float(np.mean(acc)) if acc else None,
        "f1_mean": float(np.mean(f1)) if f1 else None,
        "auroc_mean": float(np.mean(auc)) if auc else None,
    }


def maybe_get_ablation_threshold(default_threshold):
    if bool(CFG.get("sim_ablation_mode", False)):
        fixed_thr = float(CFG.get("sim_fixed_threshold", 0.5))
        print("Ablation mode: using fixed threshold =", fixed_thr)
        return fixed_thr
    return default_threshold


def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def save_model_bundle(
    federated_model=None,
    logreg_model=None,
    feature_names=None,
    metadata=None,
    example_input=None,
):
    models_dir = ensure_models_dir()

    saved = {
        "models_dir": str(models_dir),
        "federated_model_path": None,
        "logreg_model_path": None,
        "feature_names_path": None,
        "metadata_path": None,
        "example_input_path": None,
    }

    if federated_model is not None:
        federated_path = models_dir / "federated_pytorch_model.pt"
        torch.save(federated_model.state_dict(), federated_path)
        saved["federated_model_path"] = str(federated_path)

    if logreg_model is not None:
        logreg_path = models_dir / "centralized_logreg.skops"
        sio.dump(logreg_model, logreg_path)
        saved["logreg_model_path"] = str(logreg_path)

    if feature_names is not None:
        feature_names_path = models_dir / "feature_names.json"
        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump({"feature_names": list(feature_names)}, f, indent=2)
        saved["feature_names_path"] = str(feature_names_path)

    if metadata is not None:
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        saved["metadata_path"] = str(metadata_path)

    if example_input is not None:
        example_input_path = models_dir / "input_example.npy"
        np.save(example_input_path, example_input.astype(np.float32))
        saved["example_input_path"] = str(example_input_path)

    return saved


def run_leave_one_center_out(center_data, model_type):
    results = {}
    centers = list(center_data.keys())

    for test_center in centers:
        train_centers = [c for c in centers if c != test_center]

        X_train_list = []
        y_train_list = []

        for c in train_centers:
            X_train_list.append(center_data[c]["X_train"])
            y_train_list.append(center_data[c]["y_train"])

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        X_test = center_data[test_center]["X_val"]
        y_test = center_data[test_center]["y_val"]

        if model_type == "logreg":
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            threshold = 0.5

        elif model_type == "mlp":
            model = fit_mlp_model(X_train, y_train, input_dim=X_train.shape[1])
            model.eval()

            with torch.no_grad():
                train_prob = torch.softmax(
                    model(torch.tensor(X_train, dtype=torch.float32)), dim=1
                )[:, 1].cpu().numpy()

                learned_thr = choose_best_threshold(y_train, train_prob, metric="balanced_accuracy")
                threshold = maybe_get_ablation_threshold(learned_thr)

                probs = torch.softmax(
                    model(torch.tensor(X_test, dtype=torch.float32)), dim=1
                )[:, 1].cpu().numpy()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        metrics, _, _ = compute_binary_metrics(y_test, probs, threshold=threshold)
        metrics["threshold"] = float(threshold)
        results[test_center] = metrics

    return results


def run_federated_mode(train_files, val_files):
    selected_genes = get_global_selected_genes()
    center_data = {}

    for train_file in train_files:
        center_name = train_file.stem.replace("_train", "")
        val_file = PROC_DIR / f"{center_name}_val.csv"

        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        usable_genes = [g for g in selected_genes if g in train_df.columns and g in val_df.columns]

        X_train = train_df[usable_genes].values.astype(float)
        y_train = train_df["label"].values.astype(int)

        X_val = val_df[usable_genes].values.astype(float)
        y_val = val_df["label"].values.astype(int)

        X_train_z, X_val_z, _, _ = standardize_train_val(X_train, X_val)

        center_data[center_name] = {
            "X_train": X_train_z,
            "y_train": y_train,
            "X_val": X_val_z,
            "y_val": y_val,
            "feature_names": usable_genes,
        }

    input_dim = len(next(iter(center_data.values()))["feature_names"])

    global_model = None
    n_epochs = int(CFG.get("epochs", 20))
    learning_rate = float(CFG.get("learning_rate", 0.001))

    for _ in range(n_epochs):
        local_state_dicts = []

        for _, d in center_data.items():
            X_train = torch.tensor(d["X_train"], dtype=torch.float32)
            y_train = torch.tensor(d["y_train"], dtype=torch.long)

            model = GeneExpressionNet(input_dim=input_dim)
            if global_model is not None:
                model.load_state_dict(global_model.state_dict())

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=min(32, len(X_train)),
                shuffle=True,
            )

            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            local_state_dicts.append(model.state_dict())

        avg_state = federated_average(local_state_dicts)
        global_model = GeneExpressionNet(input_dim=input_dim)
        global_model.load_state_dict(avg_state)

    train_probs_all = []
    train_y_all = []

    global_model.eval()
    with torch.no_grad():
        for _, d in center_data.items():
            X_train = torch.tensor(d["X_train"], dtype=torch.float32)
            y_train = d["y_train"]

            prob = torch.softmax(global_model(X_train), dim=1)[:, 1].cpu().numpy()
            train_probs_all.append(prob)
            train_y_all.append(y_train)

    train_probs_all = np.concatenate(train_probs_all)
    train_y_all = np.concatenate(train_y_all)

    learned_threshold = choose_best_threshold(
        train_y_all,
        train_probs_all,
        metric="balanced_accuracy",
    )
    best_threshold = maybe_get_ablation_threshold(learned_threshold)

    print(f"Chosen decision threshold from pooled training data: {best_threshold:.4f}")
    print(
        "Training probability summary: min={0:.4f}, mean={1:.4f}, max={2:.4f}".format(
            train_probs_all.min(),
            train_probs_all.mean(),
            train_probs_all.max(),
        )
    )

    federated_metrics = {}
    roc_rows = []
    pred_rows = []

    with torch.no_grad():
        for center_name, d in center_data.items():
            X_val = torch.tensor(d["X_val"], dtype=torch.float32)
            y_val = d["y_val"]

            prob = torch.softmax(X_val if False else global_model(X_val), dim=1)[:, 1].cpu().numpy()

            metrics, fpr, tpr = compute_binary_metrics(
                y_val,
                prob,
                threshold=best_threshold,
            )
            metrics["threshold"] = float(best_threshold)
            federated_metrics[center_name] = metrics

            print(
                "{0}: val_label_counts={1}, prob_min={2:.4f}, prob_mean={3:.4f}, prob_max={4:.4f}".format(
                    center_name,
                    dict(pd.Series(y_val).value_counts()),
                    prob.min(),
                    prob.mean(),
                    prob.max(),
                )
            )

            for x, y in zip(fpr, tpr):
                roc_rows.append({"center": center_name, "fpr": float(x), "tpr": float(y)})

            pred_rows.append(
                pd.DataFrame(
                    {
                        "model": "federated_pytorch",
                        "center": center_name,
                        "y_true": y_val,
                        "y_prob": prob,
                        "threshold": best_threshold,
                        "y_pred": (prob >= best_threshold).astype(int),
                    }
                )
            )

    X_train_pool = np.concatenate([d["X_train"] for d in center_data.values()], axis=0)
    y_train_pool = np.concatenate([d["y_train"] for d in center_data.values()], axis=0)

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    logreg.fit(X_train_pool, y_train_pool)

    train_prob_lr = logreg.predict_proba(X_train_pool)[:, 1]
    learned_threshold_lr = choose_best_threshold(
        y_train_pool,
        train_prob_lr,
        metric="balanced_accuracy",
    )
    best_threshold_lr = maybe_get_ablation_threshold(learned_threshold_lr)

    baseline_metrics = {}

    for center_name, d in center_data.items():
        X_val = d["X_val"]
        y_val = d["y_val"]

        prob = logreg.predict_proba(X_val)[:, 1]
        metrics, _, _ = compute_binary_metrics(
            y_val,
            prob,
            threshold=best_threshold_lr,
        )
        metrics["threshold"] = float(best_threshold_lr)
        baseline_metrics[center_name] = metrics

        pred_rows.append(
            pd.DataFrame(
                {
                    "model": "centralized_logreg",
                    "center": center_name,
                    "y_true": y_val,
                    "y_prob": prob,
                    "threshold": best_threshold_lr,
                    "y_pred": (prob >= best_threshold_lr).astype(int),
                }
            )
        )

    federated_avg = compute_average_metrics(federated_metrics)
    logreg_avg = compute_average_metrics(baseline_metrics)

    loco_mlp = run_leave_one_center_out(center_data, "mlp")
    loco_lr = run_leave_one_center_out(center_data, "logreg")

    loco_mlp_avg = compute_average_metrics(loco_mlp)
    loco_lr_avg = compute_average_metrics(loco_lr)

    results = {
        "mode": "federated_multi_center",
        "federated_pytorch": federated_metrics,
        "centralized_logreg": baseline_metrics,
        "federated_average": federated_avg,
        "logreg_average": logreg_avg,
        "loco_federated_pytorch": loco_mlp,
        "loco_centralized_logreg": loco_lr,
        "loco_federated_average": loco_mlp_avg,
        "loco_logreg_average": loco_lr_avg,
    }

    with open(PROC_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(roc_rows).to_csv(PROC_DIR / "roc_points.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(PROC_DIR / "predictions.csv", index=False)

    with open(PROC_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "federated_pytorch_threshold": float(best_threshold),
                "centralized_logreg_threshold": float(best_threshold_lr),
            },
            f,
            indent=2,
        )

    model_metadata = {
        "mode": "federated_multi_center",
        "model_backend": ARGS.model_backend,
        "input_dim": input_dim,
        "n_selected_genes": len(selected_genes),
        "selected_genes": selected_genes,
        "federated_threshold": float(best_threshold),
        "logreg_threshold": float(best_threshold_lr),
        "federated_average_auroc": federated_avg.get("auroc_mean"),
        "logreg_average_auroc": logreg_avg.get("auroc_mean"),
    }

    save_model_bundle(
        federated_model=global_model,
        logreg_model=logreg,
        feature_names=selected_genes,
        metadata=model_metadata,
        example_input=X_train_pool[:1],
    )

    print(json.dumps(results, indent=2))
    print("Federated training complete")

    if os.getenv("FEDOMICS_MLFLOW_MANAGED_BY_PIPELINE") != "1":
        log_standalone_training_run(
            run_name="train_federated_standalone",
            params={"model_backend": ARGS.model_backend, "mode": "federated_multi_center"},
            config_path=str(CFG_PATH),
            processed_dir=PROC_DIR,
        )


def find_single_real_center_file():
    raw_center_files = sorted(RAW_TCGA_CENTERS_DIR.glob("center_*_expression.csv"))
    if len(raw_center_files) == 1:
        return raw_center_files[0]
    return None


def run_single_cohort_cv_mode():
    print("Switching to exploratory single-cohort repeated CV mode.")

    raw_center_file = find_single_real_center_file()
    if raw_center_file is None:
        raise FileNotFoundError(
            "Expected exactly one raw TCGA center file in data/raw/tcga_prad/centers for single-cohort CV mode."
        )

    full_df = pd.read_csv(raw_center_file).drop_duplicates().reset_index(drop=True)

    if "label" not in full_df.columns:
        raise ValueError("Raw center dataframe must contain a 'label' column.")

    y = full_df["label"].values.astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Single-cohort CV mode requires both classes.")

    print("Single-cohort CV source:", raw_center_file)
    print("Single-cohort CV total real samples:", len(full_df))
    print("Single-cohort CV label counts:", dict(pd.Series(y).value_counts()))

    splitter = StratifiedShuffleSplit(
        n_splits=int(CFG.get("tcga_single_cohort_cv_splits", 5)),
        test_size=float(CFG.get("tcga_single_cohort_cv_test_size", 0.25)),
        random_state=42,
    )

    top_k = int(CFG.get("tcga_cv_top_k", 8))

    logreg_metrics_all = []
    mlp_metrics_all = []
    pred_rows = []
    selected_genes_by_split = []

    final_logreg_model = None
    final_mlp_model = None
    final_feature_set = None
    final_example_input = None

    for split_idx, (tr_idx, va_idx) in enumerate(splitter.split(full_df.drop(columns=["label"]), y), start=1):
        train_df = full_df.iloc[tr_idx].reset_index(drop=True)
        val_df = full_df.iloc[va_idx].reset_index(drop=True)

        split_selected_genes = select_top_k_features_train_only(train_df, top_k=top_k)
        selected_genes_by_split.append(
            {
                "split": split_idx,
                "selected_genes": split_selected_genes,
            }
        )

        X_train = train_df[split_selected_genes].values.astype(float)
        y_train = train_df["label"].values.astype(int)

        X_val = val_df[split_selected_genes].values.astype(float)
        y_val = val_df["label"].values.astype(int)

        X_train_z, X_val_z, _, _ = standardize_train_val(X_train, X_val)

        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        logreg.fit(X_train_z, y_train)

        train_prob_lr = logreg.predict_proba(X_train_z)[:, 1]
        thr_lr = choose_best_threshold(y_train, train_prob_lr, metric="balanced_accuracy")

        val_prob_lr = logreg.predict_proba(X_val_z)[:, 1]
        lr_metrics, _, _ = compute_binary_metrics(y_val, val_prob_lr, threshold=thr_lr)
        lr_metrics["threshold"] = float(thr_lr)
        logreg_metrics_all.append(lr_metrics)

        pred_rows.append(
            pd.DataFrame(
                {
                    "model": "centralized_logreg_cv",
                    "split": split_idx,
                    "y_true": y_val,
                    "y_prob": val_prob_lr,
                    "threshold": thr_lr,
                    "y_pred": (val_prob_lr >= thr_lr).astype(int),
                }
            )
        )

        mlp = fit_mlp_model(X_train_z, y_train, input_dim=len(split_selected_genes))
        mlp.eval()

        with torch.no_grad():
            train_prob_mlp = torch.softmax(
                mlp(torch.tensor(X_train_z, dtype=torch.float32)), dim=1
            )[:, 1].cpu().numpy()

            thr_mlp = choose_best_threshold(y_train, train_prob_mlp, metric="balanced_accuracy")

            val_prob_mlp = torch.softmax(
                mlp(torch.tensor(X_val_z, dtype=torch.float32)), dim=1
            )[:, 1].cpu().numpy()

        mlp_metrics, _, _ = compute_binary_metrics(y_val, val_prob_mlp, threshold=thr_mlp)
        mlp_metrics["threshold"] = float(thr_mlp)
        mlp_metrics_all.append(mlp_metrics)

        pred_rows.append(
            pd.DataFrame(
                {
                    "model": "centralized_mlp_cv",
                    "split": split_idx,
                    "y_true": y_val,
                    "y_prob": val_prob_mlp,
                    "threshold": thr_mlp,
                    "y_pred": (val_prob_mlp >= thr_mlp).astype(int),
                }
            )
        )

        final_logreg_model = logreg
        final_mlp_model = mlp
        final_feature_set = split_selected_genes
        final_example_input = X_train_z[:1]

        print(
            "Split {0}: y_val_counts={1}, n_features={2}, logreg_auroc={3}, mlp_auroc={4}".format(
                split_idx,
                dict(pd.Series(y_val).value_counts()),
                len(split_selected_genes),
                lr_metrics.get("auroc"),
                mlp_metrics.get("auroc"),
            )
        )

    results = {
        "mode": "single_cohort_repeated_cv",
        "centralized_logreg_cv": summarize_metric_dicts(logreg_metrics_all),
        "centralized_mlp_cv": summarize_metric_dicts(mlp_metrics_all),
        "notes": [
            "Evaluation used the raw real TCGA center file only.",
            "No augmented synthetic rows were included in the pooled CV dataset.",
            "Feature selection was performed inside each CV split using training data only.",
            "Real cohort was too small for meaningful federated evaluation.",
            "Reported results are repeated stratified CV summaries on a single pooled cohort.",
        ],
    }

    with open(PROC_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.concat(pred_rows, ignore_index=True).to_csv(PROC_DIR / "predictions.csv", index=False)

    with open(PROC_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"note": "thresholds vary by CV split and model"}, f, indent=2)

    pd.DataFrame().to_csv(PROC_DIR / "roc_points.csv", index=False)

    selected_rows = []
    for item in selected_genes_by_split:
        split_id = item["split"]
        for g in item["selected_genes"]:
            selected_rows.append({"split": split_id, "selected_gene": g})
    pd.DataFrame(selected_rows).to_csv(PROC_DIR / "selected_genes_cv.csv", index=False)

    model_metadata = {
        "mode": "single_cohort_repeated_cv",
        "model_backend": ARGS.model_backend,
        "n_splits": int(CFG.get("tcga_single_cohort_cv_splits", 5)),
        "top_k": top_k,
        "last_split_feature_count": len(final_feature_set) if final_feature_set else None,
        "logreg_cv_auroc_mean": results["centralized_logreg_cv"].get("auroc_mean"),
        "mlp_cv_auroc_mean": results["centralized_mlp_cv"].get("auroc_mean"),
    }

    save_model_bundle(
        federated_model=final_mlp_model,
        logreg_model=final_logreg_model,
        feature_names=final_feature_set,
        metadata=model_metadata,
        example_input=final_example_input,
    )

    print(json.dumps(results, indent=2))
    print("Single-cohort repeated CV evaluation complete")

    if os.getenv("FEDOMICS_MLFLOW_MANAGED_BY_PIPELINE") != "1":
        log_standalone_training_run(
            run_name="train_federated_single_cohort_cv_standalone",
            params={"model_backend": ARGS.model_backend, "mode": "single_cohort_repeated_cv"},
            config_path=str(CFG_PATH),
            processed_dir=PROC_DIR,
        )


if len(train_files) == 1:
    run_single_cohort_cv_mode()
else:
    run_federated_mode(train_files, val_files)