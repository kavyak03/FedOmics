from pathlib import Path
import sys
import re
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_yaml

CFG = load_yaml("configs/config.yaml")
RNG = np.random.default_rng(CFG.get("random_seed", 42))

RAW_DIR = Path("data/demo_dataset")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_tcga_case_id(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s)
    if m:
        return m.group(1)
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return None


def _find_case_column(columns):
    preferred = [
        "cases.submitter_id",
        "case_submitter_id",
        "submitter_id",
        "cases.case_id",
        "case_id",
    ]
    for p in preferred:
        if p in columns:
            return p
    for c in columns:
        if "submitter" in c.lower():
            return c
    for c in columns:
        if "case" in c.lower():
            return c
    return None


def _find_gleason_column(columns):
    preferred = [
        "diagnoses.gleason_score",
        "diagnoses.gleason_grade_group",
        "diagnoses.gleason_primary_pattern",
        "diagnoses.gleason_secondary_pattern",
    ]
    for p in preferred:
        if p in columns:
            return p
    for c in columns:
        if "gleason" in c.lower():
            return c
    return None


def _parse_gleason_to_binary_label(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    m_sum = re.search(r"(\d+)\s*\+\s*(\d+)", s)
    if m_sum:
        total = int(m_sum.group(1)) + int(m_sum.group(2))
        return 1 if total >= 8 else 0

    m_num = re.search(r"(\d+)", s)
    if m_num:
        score = int(m_num.group(1))
        return 1 if score >= 8 else 0

    return None


def load_tcga_reference_labeled_dataframe():
    expr_path = Path(CFG.get("sim_reference_expression_path", "data/raw/tcga_prad/expression_matrix.csv"))
    clin_path = Path(CFG.get("sim_reference_clinical_path", "data/raw/tcga_prad/clinical.tsv"))

    if not expr_path.exists() or not clin_path.exists():
        return None

    expr = pd.read_csv(expr_path)
    clinical = pd.read_csv(clin_path, sep="\t")
    clinical.columns = [c.lower() for c in clinical.columns]

    case_col = _find_case_column(list(clinical.columns))
    gleason_col = _find_gleason_column(list(clinical.columns))

    if case_col is None or gleason_col is None:
        return None

    clinical = clinical[[case_col, gleason_col]].dropna().copy()
    clinical["case_short"] = clinical[case_col].apply(_normalize_tcga_case_id)
    clinical["label"] = clinical[gleason_col].apply(_parse_gleason_to_binary_label)
    clinical = clinical.dropna(subset=["case_short", "label"]).copy()
    clinical["label"] = clinical["label"].astype(int)
    clinical = clinical.drop_duplicates(subset=["case_short"])

    sample_cols = [c for c in expr.columns if c != "gene"]
    mat = expr[sample_cols].T
    mat.columns = expr["gene"].astype(str).tolist()
    mat.index.name = "sample_id"
    mat = mat.reset_index()

    mat["case_short"] = mat["sample_id"].apply(_normalize_tcga_case_id)
    merged = mat.merge(clinical[["case_short", "label"]], on="case_short", how="inner")
    merged = merged.drop(columns=["sample_id", "case_short"])

    if merged.empty:
        return None

    feature_cols = [c for c in merged.columns if c != "label"]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    merged["label"] = merged["label"].astype(int)

    return merged


def build_generic_gene_names(n_genes):
    return ["GENE_{0:04d}".format(i + 1) for i in range(n_genes)]


def build_module_membership(n_genes, num_modules, module_size):
    all_idx = np.arange(n_genes)
    RNG.shuffle(all_idx)

    modules = []
    cursor = 0
    for _ in range(num_modules):
        end = min(cursor + module_size, n_genes)
        modules.append(list(all_idx[cursor:end]))
        cursor = end
        if cursor >= n_genes:
            break
    return modules


def _draw_label_from_signal(signal_mode, linear_score, interaction_score):
    if signal_mode == "linear":
        disease_score = linear_score + RNG.normal(0.0, 0.75)

    elif signal_mode == "interaction":
        disease_score = 1.6 * interaction_score + 0.05 * linear_score + RNG.normal(0.0, 0.55)

    elif signal_mode == "mixed":
        disease_score = 0.40 * linear_score + 1.10 * interaction_score + RNG.normal(0.0, 0.60)

    else:
        raise ValueError("Unknown signal_mode: {0}".format(signal_mode))

    y = int(disease_score > 0.0)
    return y, disease_score


def simulate_generic_coexpression_center(center_id, n_samples, signal_mode):
    n_genes = int(CFG.get("sim_total_genes", 250))
    num_modules = int(CFG.get("sim_num_modules", 8))
    num_signal_modules = int(CFG.get("sim_num_signal_modules", 3))
    module_size = int(CFG.get("sim_module_size", 25))

    signal_strength = float(CFG.get("sim_signal_strength", 1.0))
    noise_scale = float(CFG.get("sim_noise_scale", 0.60))
    center_shift_scale = float(CFG.get("sim_center_shift_scale", 0.08))

    genes = build_generic_gene_names(n_genes)
    modules = build_module_membership(n_genes, num_modules, module_size)

    if len(modules) < 3:
        raise ValueError("Need at least 3 modules for linear/interaction/mixed simulator.")

    signal_module_ids = list(range(min(num_signal_modules, len(modules))))

    gene_baseline = RNG.normal(loc=5.0, scale=0.6, size=n_genes)
    center_shift = RNG.normal(0.0, center_shift_scale, size=n_genes)

    module_loadings = []
    module_noise_scales = []
    for module in modules:
        loadings = RNG.normal(loc=1.0, scale=0.18, size=len(module))
        module_loadings.append(loadings)
        module_noise_scales.append(float(abs(RNG.normal(1.0, 0.12))))

    rows = []

    for _ in range(n_samples):
        sample_global_shift = RNG.normal(0.0, 0.22)
        x = gene_baseline.copy() + center_shift + sample_global_shift

        module_activity = RNG.normal(0.0, 1.0, size=len(modules))

        a = module_activity[signal_module_ids[0]]
        b = module_activity[signal_module_ids[1]] if len(signal_module_ids) > 1 else 0.0
        c = module_activity[signal_module_ids[2]] if len(signal_module_ids) > 2 else 0.0

        linear_score = 1.0 * a + 0.8 * b - 0.6 * c
        interaction_score = (a * b) - 0.20 * (b * c)

        y, _ = _draw_label_from_signal(
            signal_mode=signal_mode,
            linear_score=linear_score,
            interaction_score=interaction_score,
        )

        for m_id, module in enumerate(modules):
            loadings = module_loadings[m_id]
            local_noise_scale = module_noise_scales[m_id]

            for j, gene_idx in enumerate(module):
                x[gene_idx] += module_activity[m_id] * loadings[j]

                if signal_mode == "linear":
                    if m_id in signal_module_ids:
                        x[gene_idx] += signal_strength * linear_score * loadings[j] * 0.48

                elif signal_mode == "interaction":
                    if m_id == signal_module_ids[0]:
                        x[gene_idx] += signal_strength * (a * b) * loadings[j] * 0.52
                    elif len(signal_module_ids) > 1 and m_id == signal_module_ids[1]:
                        x[gene_idx] += signal_strength * (a * b) * loadings[j] * 0.52
                    elif len(signal_module_ids) > 2 and m_id == signal_module_ids[2]:
                        x[gene_idx] += signal_strength * (b * c) * loadings[j] * -0.12

                elif signal_mode == "mixed":
                    if m_id in signal_module_ids:
                        x[gene_idx] += signal_strength * linear_score * loadings[j] * 0.12

                    if m_id == signal_module_ids[0]:
                        x[gene_idx] += signal_strength * (a * b) * loadings[j] * 0.34
                    elif len(signal_module_ids) > 1 and m_id == signal_module_ids[1]:
                        x[gene_idx] += signal_strength * (a * b) * loadings[j] * 0.34
                    elif len(signal_module_ids) > 2 and m_id == signal_module_ids[2]:
                        x[gene_idx] += signal_strength * (b * c) * loadings[j] * -0.10

                x[gene_idx] += RNG.normal(0.0, noise_scale * local_noise_scale)

        if signal_mode == "linear":
            label_shift = 0.16 if y == 1 else -0.16
            for m_id in signal_module_ids:
                for gene_idx in modules[m_id]:
                    x[gene_idx] += label_shift

        elif signal_mode == "mixed":
            label_shift = 0.03 if y == 1 else -0.03
            for m_id in signal_module_ids:
                for gene_idx in modules[m_id]:
                    x[gene_idx] += label_shift

        x = np.maximum(x, 0.0)

        row = dict(zip(genes, x))
        row["label"] = y
        row["signal_mode"] = signal_mode
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "center_{0}_expression.csv".format(center_id), index=False)


def _compute_tcga_class_stats(ref_df):
    feature_cols = [c for c in ref_df.columns if c != "label"]
    X = ref_df[feature_cols].values.astype(float)
    y = ref_df["label"].values.astype(int)

    global_mean = X.mean(axis=0)

    class_stats = {}
    for label in sorted(np.unique(y)):
        Xc = X[y == label]
        class_mean = Xc.mean(axis=0)
        class_stats[int(label)] = {
            "mean": class_mean,
            "n": int(len(Xc)),
        }

    return feature_cols, global_mean, class_stats, X


def simulate_tcga_matched_center(center_id, n_samples, ref_df, signal_mode):
    """
    TCGA-informed simulation with optional label-conditional class anchoring.
    This is the important fix for mixed-mode TCGA-informed simulation.
    """
    feature_cols, global_mean, class_stats, real_X = _compute_tcga_class_stats(ref_df)

    max_cov_genes = int(CFG.get("sim_cov_max_genes", 500))
    if real_X.shape[1] > max_cov_genes:
        gene_vars = real_X.var(axis=0)
        top_idx = np.argsort(gene_vars)[-max_cov_genes:]
        real_X = real_X[:, top_idx]
        feature_cols = [feature_cols[i] for i in top_idx]
        global_mean = global_mean[top_idx]

        for label in class_stats:
            class_stats[label]["mean"] = class_stats[label]["mean"][top_idx]

    cov = np.cov(real_X, rowvar=False)
    diag = np.diag(np.diag(cov))
    cov = 0.9 * cov + 0.1 * diag
    cov = cov + np.eye(cov.shape[0]) * 1e-6

    signal_strength = float(CFG.get("sim_signal_strength", 1.0))
    noise_scale = float(CFG.get("sim_noise_scale", 0.60))
    center_shift_scale = float(CFG.get("sim_center_shift_scale", 0.08))

    center_shift = RNG.normal(0.0, center_shift_scale, size=len(feature_cols))
    gene_vars = real_X.var(axis=0)

    order = np.argsort(gene_vars)[::-1]
    block = max(10, min(25, len(order) // 10))
    idx_a = list(order[0:block])
    idx_b = list(order[block:2 * block])
    idx_c = list(order[2 * block:3 * block])

    use_label_conditional = bool(CFG.get("sim_tcga_label_conditional", True))
    blend_weight = float(CFG.get("sim_tcga_class_blend_weight", 0.75))

    rows = []
    for _ in range(n_samples):
        # Draw latent module states first
        a = RNG.normal(0.0, 1.0)
        b = RNG.normal(0.0, 1.0)
        c = RNG.normal(0.0, 1.0)

        linear_score = 1.0 * a + 0.8 * b - 0.6 * c
        interaction_score = (a * b) - 0.20 * (b * c)

        # Choose class from synthetic rule
        y, _ = _draw_label_from_signal(
            signal_mode=signal_mode,
            linear_score=linear_score,
            interaction_score=interaction_score,
        )

        # IMPORTANT FIX:
        # anchor the sample to the correct TCGA class background
        if use_label_conditional and y in class_stats:
            class_mean = class_stats[y]["mean"]
            sim_mean = blend_weight * class_mean + (1.0 - blend_weight) * global_mean
        else:
            sim_mean = global_mean

        sample_global_shift = RNG.normal(0.0, 0.18)
        x = RNG.multivariate_normal(
            mean=sim_mean + center_shift + sample_global_shift,
            cov=cov * noise_scale,
            check_valid="ignore",
        )

        if signal_mode == "linear":
            for idx in idx_a:
                x[idx] += signal_strength * linear_score * 0.30
            for idx in idx_b:
                x[idx] += signal_strength * linear_score * 0.22

        elif signal_mode == "interaction":
            for idx in idx_a:
                x[idx] += signal_strength * (a * b) * 0.34
            for idx in idx_b:
                x[idx] += signal_strength * (a * b) * 0.34
            for idx in idx_c:
                x[idx] += signal_strength * (b * c) * -0.10

        elif signal_mode == "mixed":
            # reduced additive, stronger interaction, but still class-consistent
            for idx in idx_a:
                x[idx] += signal_strength * linear_score * 0.10
                x[idx] += signal_strength * (a * b) * 0.20
            for idx in idx_b:
                x[idx] += signal_strength * linear_score * 0.08
                x[idx] += signal_strength * (a * b) * 0.20
            for idx in idx_c:
                x[idx] += signal_strength * (b * c) * -0.08

        x = np.maximum(x, 0.0)

        row = dict(zip(feature_cols, x))
        row["label"] = y
        row["signal_mode"] = signal_mode
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "center_{0}_expression.csv".format(center_id), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal-mode",
        choices=["linear", "interaction", "mixed"],
        default="mixed",
    )
    args = parser.parse_args()

    for f in RAW_DIR.glob("center_*_expression.csv"):
        f.unlink()

    n_centers = int(CFG.get("num_centers", 3))
    samples_per_center = int(CFG.get("samples_per_center", 200))
    sim_generator_mode = str(CFG.get("sim_generator_mode", "generic_coexpression")).lower()
    signal_mode = args.signal_mode

    if sim_generator_mode == "tcga_matched":
        ref_df = load_tcga_reference_labeled_dataframe()
        if ref_df is not None:
            print("Using TCGA-matched simulation mode (signal_mode='{0}').".format(signal_mode))
            print("TCGA label-conditional simulation enabled:", bool(CFG.get("sim_tcga_label_conditional", True)))
            for c in range(1, n_centers + 1):
                simulate_tcga_matched_center(c, samples_per_center, ref_df, signal_mode=signal_mode)
            print(
                "Simulated dataset generated for {0} centers with {1} samples per center".format(
                    n_centers, samples_per_center
                )
            )
            return
        else:
            if bool(CFG.get("sim_fallback_to_generic", True)):
                print(
                    "TCGA reference not available or not usable. "
                    "Falling back to generic coexpression simulation (signal_mode='{0}').".format(signal_mode)
                )
            else:
                raise ValueError("TCGA-matched simulation requested but reference data could not be loaded.")

    print("Using generic coexpression-aware simulation mode (signal_mode='{0}').".format(signal_mode))
    for c in range(1, n_centers + 1):
        simulate_generic_coexpression_center(c, samples_per_center, signal_mode=signal_mode)

    print(
        "Simulated dataset generated for {0} centers with {1} samples per center".format(
            n_centers, samples_per_center
        )
    )


if __name__ == "__main__":
    main() 