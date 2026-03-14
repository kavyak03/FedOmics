from pathlib import Path
import json
import shutil
import subprocess
import argparse
import sys
import re

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_yaml

CFG = load_yaml("configs/config.yaml")
RNG = np.random.default_rng(CFG.get("random_seed", 42))

BASE = Path("data/raw/tcga_prad")
FILES_DIR = BASE / "files"
CENTERS_DIR = BASE / "centers"

BASE.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)
CENTERS_DIR.mkdir(parents=True, exist_ok=True)

FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


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


def _extract_sample_submitter_id(hit):
    """
    Try to recover a sample-level submitter ID from the nested GDC structure.
    Prefer a Primary Tumor sample when possible.
    """
    cases = hit.get("cases", [])
    for case in cases:
        samples = case.get("samples", [])
        if not samples:
            continue

        # Prefer Primary Tumor if present
        for sample in samples:
            sample_type = str(sample.get("sample_type", "")).strip().lower()
            submitter_id = sample.get("submitter_id")
            if submitter_id and sample_type == "primary tumor":
                return str(submitter_id)

        # Otherwise return first available sample submitter id
        for sample in samples:
            submitter_id = sample.get("submitter_id")
            if submitter_id:
                return str(submitter_id)

    return None


def _extract_case_submitter_id(hit):
    cases = hit.get("cases", [])
    if cases:
        submitter_id = cases[0].get("submitter_id")
        if submitter_id:
            return str(submitter_id)
    return None


def _extract_case_id(hit):
    cases = hit.get("cases", [])
    if cases:
        case_id = cases[0].get("case_id")
        if case_id:
            return str(case_id)
    return None


def build_manifest():
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-PRAD"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                    "value": ["open"],
                },
            },
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "md5sum",
                "file_size",
                "cases.submitter_id",
                "cases.case_id",
                "cases.samples.submitter_id",
                "cases.samples.sample_type",
            ]
        ),
        "format": "JSON",
        "size": "500",
    }

    resp = requests.get(FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for hit in hits:
        rows.append(
            {
                "file_id": hit.get("file_id"),
                "file_name": hit.get("file_name"),
                "md5": hit.get("md5sum"),
                "file_size": hit.get("file_size"),
                "sample_submitter_id": _extract_sample_submitter_id(hit),
                "case_submitter_id": _extract_case_submitter_id(hit),
                "case_id": _extract_case_id(hit),
            }
        )

    manifest = pd.DataFrame(rows).dropna(subset=["file_id"])
    manifest.to_csv(BASE / "manifest.tsv", sep="\t", index=False)
    print("Manifest written:", BASE / "manifest.tsv")
    return manifest


def download_via_api(manifest, max_files=500):
    for _, row in manifest.head(max_files).iterrows():
        file_id = row["file_id"]
        file_name = row["file_name"] if pd.notna(row["file_name"]) else "{0}.tsv".format(file_id)

        out_dir = FILES_DIR / str(file_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / str(file_name)

        if out_path.exists():
            continue

        resp = requests.get("{0}/{1}".format(DATA_ENDPOINT, file_id), timeout=120)

        if resp.status_code == 200:
            out_path.write_bytes(resp.content)
            print("Downloaded", file_name)
        else:
            print("Skipping", file_id, "HTTP", resp.status_code)


def download_via_client():
    manifest_path = BASE / "manifest.tsv"

    if shutil.which("gdc-client") is None:
        raise RuntimeError(
            "gdc-client not found on PATH. Use --download-mode api or install gdc-client."
        )

    subprocess.run(
        ["gdc-client", "download", "-m", str(manifest_path), "-d", str(FILES_DIR)],
        check=True,
    )


def parse_count_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            gene = parts[0].strip()
            if not gene:
                continue

            try:
                value = float(parts[-1])
            except ValueError:
                continue

            rows.append((gene, value))

    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows, columns=["gene", path.stem])
    df = df.drop_duplicates(subset=["gene"])
    return df


def build_expression_matrix():
    manifest_path = BASE / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.tsv not found. Run build_manifest() first.")

    manifest = pd.read_csv(manifest_path, sep="\t")

    file_to_sample = {}
    file_to_case = {}

    for _, row in manifest.iterrows():
        file_id = str(row["file_id"])

        sample_submitter_id = row.get("sample_submitter_id", None)
        case_submitter_id = row.get("case_submitter_id", None)

        if pd.notna(sample_submitter_id):
            file_to_sample[file_id] = str(sample_submitter_id)

        if pd.notna(case_submitter_id):
            file_to_case[file_id] = str(case_submitter_id)

    candidates = list(FILES_DIR.rglob("*.tsv")) + list(FILES_DIR.rglob("*.txt"))
    if not candidates:
        raise FileNotFoundError("No count files found under {0}".format(FILES_DIR))

    merged = None
    parsed_files = 0
    mapped_file_ids = []

    for fp in candidates:
        one = parse_count_file(fp)
        if one is None:
            continue

        file_id = fp.parent.name

        # Prefer sample-level ID, fallback to case-level ID
        expr_id = file_to_sample.get(file_id, None)
        if expr_id is None:
            expr_id = file_to_case.get(file_id, None)

        if expr_id is None:
            continue

        old_col = one.columns[1]
        one = one.rename(columns={old_col: expr_id})

        merged = one if merged is None else merged.merge(one, on="gene", how="outer")
        parsed_files += 1
        mapped_file_ids.append(file_id)

    if merged is None:
        raise ValueError("No parsable count files found that could be mapped to TCGA IDs")

    merged = merged.fillna(0.0)

    # Average duplicate sample columns if the same ID appears more than once
    if merged.columns.duplicated().any():
        gene_col = merged["gene"]
        value_df = merged.drop(columns=["gene"])
        value_df = value_df.groupby(level=0, axis=1).mean()
        merged = pd.concat([gene_col, value_df], axis=1)

    merged.to_csv(BASE / "expression_matrix.csv", index=False)
    print("Merged expression matrix created using {0} mapped files".format(parsed_files))
    return merged


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


def extract_labels(sample_ids):
    clinical_path = BASE / "clinical.tsv"
    if not clinical_path.exists():
        raise FileNotFoundError(
            "TCGA clinical file missing. Put clinical.tsv in data/raw/tcga_prad/."
        )

    clinical = pd.read_csv(clinical_path, sep="\t")
    clinical.columns = [c.lower() for c in clinical.columns]

    case_col = _find_case_column(list(clinical.columns))
    gleason_col = _find_gleason_column(list(clinical.columns))

    if case_col is None:
        raise ValueError("Could not find a case ID column in clinical.tsv")
    if gleason_col is None:
        raise ValueError("Could not find a Gleason-related column in clinical.tsv")

    print("Using clinical case column:", case_col)
    print("Using Gleason column:", gleason_col)

    subset = clinical[[case_col, gleason_col]].dropna().copy()
    subset["case_short"] = subset[case_col].apply(_normalize_tcga_case_id)
    subset["label"] = subset[gleason_col].apply(_parse_gleason_to_binary_label)

    subset = subset.dropna(subset=["case_short", "label"]).copy()
    subset["label"] = subset["label"].astype(int)
    subset = subset.drop_duplicates(subset=["case_short"])

    expr_df = pd.DataFrame({"sample_id": sample_ids})
    expr_df["sample_id"] = expr_df["sample_id"].astype(str)
    expr_df["case_short"] = expr_df["sample_id"].apply(_normalize_tcga_case_id)

    unique_expr_samples = expr_df["sample_id"].nunique()
    unique_expr_cases = expr_df["case_short"].dropna().nunique()
    unique_clinical_cases = subset["case_short"].dropna().nunique()

    expr_case_set = set(expr_df["case_short"].dropna().unique())
    clinical_case_set = set(subset["case_short"].dropna().unique())
    overlap_cases = expr_case_set.intersection(clinical_case_set)

    merged = expr_df.merge(subset[["case_short", "label"]], on="case_short", how="left")

    matched_sample_rows = merged["label"].notna().sum()
    matched_unique_cases = merged.loc[merged["label"].notna(), "case_short"].nunique()

    print("Expression samples (unique IDs):", unique_expr_samples)
    print("Expression samples with normalized TCGA case IDs:", expr_df["case_short"].notna().sum())
    print("Unique normalized expression case IDs:", unique_expr_cases)
    print("Clinical rows with usable Gleason labels:", len(subset))
    print("Unique clinical case IDs with labels:", unique_clinical_cases)
    print("Expression/clinical case overlap (unique cases):", len(overlap_cases))
    print("Matched labeled sample rows:", matched_sample_rows)
    print("Matched labeled unique cases:", matched_unique_cases)

    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    return merged[["sample_id", "label"]]


def expression_to_dataframe():
    expr = pd.read_csv(BASE / "expression_matrix.csv")
    sample_cols = [c for c in expr.columns if c != "gene"]

    mat = expr[sample_cols].T
    mat.columns = expr["gene"].tolist()
    mat.index.name = "sample_id"
    mat = mat.reset_index()

    labels = extract_labels(mat["sample_id"].tolist())
    merged = mat.merge(labels, on="sample_id", how="inner")
    merged = merged.drop(columns=["sample_id"])

    feature_cols = [c for c in merged.columns if c != "label"]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    merged["label"] = merged["label"].astype(int)

    return merged


def choose_center_count(total_real):
    if total_real < 30:
        return 1
    if total_real < 60:
        return 2
    return int(CFG.get("num_centers", 3))


def assign_real_cases_to_centers(real_df):
    n_centers = choose_center_count(len(real_df))
    print("Adaptive center count selected for TCGA mode:", n_centers)

    centers = [pd.DataFrame(columns=real_df.columns) for _ in range(n_centers)]

    for label, group in real_df.groupby("label"):
        shuffled = group.sample(
            frac=1.0,
            random_state=CFG.get("random_seed", 42)
        ).reset_index(drop=True)

        splits = np.array_split(shuffled, n_centers)

        for i, part in enumerate(splits):
            centers[i] = pd.concat([centers[i], part], ignore_index=True)

    final_centers = []
    for c in centers:
        c = c.sample(
            frac=1.0,
            random_state=CFG.get("random_seed", 42)
        ).reset_index(drop=True)
        final_centers.append(c)

    return final_centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-mode", choices=["api", "client"], default="api")
    parser.add_argument("--max-files", type=int, default=500)
    args = parser.parse_args()

    manifest = build_manifest()

    if args.download_mode == "api":
        download_via_api(manifest, max_files=args.max_files)
    else:
        download_via_client()

    build_expression_matrix()

    expr_preview = pd.read_csv(BASE / "expression_matrix.csv", nrows=2)
    expr_sample_cols = [c for c in expr_preview.columns if c != "gene"]
    print("Expression matrix sample columns (preview):", expr_sample_cols[:5])

    real_df = expression_to_dataframe()

    if real_df.empty:
        clinical_path = BASE / "clinical.tsv"
        if clinical_path.exists():
            clinical = pd.read_csv(clinical_path, sep="\t")
            print("clinical.tsv columns:", list(clinical.columns)[:20])

        raise ValueError(
            "No TCGA samples matched clinical labels. "
            "This usually means the selected Gleason column is empty/unusable "
            "or the expression IDs still do not overlap clinical case IDs."
        )

    real_counts = real_df["label"].value_counts().to_dict()
    total_real = len(real_df)

    min_real_warn = int(CFG.get("minimum_real_samples_warn", 30))
    min_class_warn = int(CFG.get("minimum_class_samples_warn", 5))

    if total_real < min_real_warn:
        print(
            "WARNING: only {0} labeled real TCGA samples available (< {1}).".format(
                total_real, min_real_warn
            )
        )

    if real_counts and min(real_counts.values()) < min_class_warn:
        print(
            "WARNING: at least one real class has fewer than {0} samples: {1}".format(
                min_class_warn, real_counts
            )
        )

    print("Real TCGA samples available before any augmentation:", total_real)
    print("Real class counts:", real_counts)
    print("NOTE: No synthetic samples are created in download_tcga.py.")
    print("Augmentation, if needed, should happen later on training folds only.")

    # Remove stale center files from older runs
    for old_file in CENTERS_DIR.glob("center_*_expression.csv"):
        old_file.unlink()

    centers = assign_real_cases_to_centers(real_df)

    for i, df in enumerate(centers, start=1):
        out = CENTERS_DIR / "center_{0}_expression.csv".format(i)
        df.to_csv(out, index=False)
        print(
            "Wrote", out,
            "with label counts",
            df["label"].value_counts().to_dict()
        )


if __name__ == "__main__":
    main()