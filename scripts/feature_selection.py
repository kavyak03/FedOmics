from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_yaml

CFG = load_yaml("configs/config.yaml")
PROC_DIR = Path("data/processed")
GENE_SET_PATH = Path("data/gene_sets/prad_pathway_genes.txt")
MAP_PATH = Path("data/gene_sets/ensembl_to_symbol.csv")


def strip_ensembl_version(gene_id):
    s = str(gene_id)
    return s.split(".")[0]


def maybe_map_ensembl_to_symbol(gene_ids):
    stripped = [strip_ensembl_version(g) for g in gene_ids]

    if not MAP_PATH.exists():
        return stripped

    mapping = pd.read_csv(MAP_PATH)
    if "ensembl_id" not in mapping.columns or "gene_symbol" not in mapping.columns:
        print("WARNING: ensembl_to_symbol.csv exists but missing required columns.")
        return stripped

    mapping["ensembl_id"] = mapping["ensembl_id"].astype(str).apply(strip_ensembl_version)
    lookup = dict(zip(mapping["ensembl_id"], mapping["gene_symbol"]))

    out = []
    for gid in stripped:
        out.append(lookup.get(gid, gid))
    return out


if not GENE_SET_PATH.exists():
    GENE_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENE_SET_PATH.write_text(
        "AR\nTP53\nBRCA1\nBRCA2\nPTEN\nMYC\nMKI67\nAKT1\nFOXA1\nERG\n",
        encoding="utf-8"
    )

train_files = sorted(PROC_DIR.glob("center_*_train.csv"))
if not train_files:
    raise FileNotFoundError("Run preprocess_data.py first.")

merged = pd.concat([pd.read_csv(f) for f in train_files], axis=0, ignore_index=True)

if "label" not in merged.columns:
    raise ValueError("Processed training data must contain a 'label' column.")

y = merged["label"]

# Keep only numeric feature columns; drop metadata/string columns
metadata_cols = {"label", "center", "signal_mode", "model_backend", "dataset_mode"}
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in metadata_cols and c != "label"]

if not feature_cols:
    raise ValueError("No numeric feature columns available for feature selection.")

X = merged[feature_cols]

original_gene_ids = list(X.columns)
display_gene_ids = maybe_map_ensembl_to_symbol(original_gene_ids)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi_scores, _ = chi2(X_scaled, y)
score_df = pd.DataFrame(
    {
        "gene": original_gene_ids,
        "display_gene": display_gene_ids,
        "chi2_score": chi_scores,
    }
).sort_values("chi2_score", ascending=False)

top_k = min(int(CFG.get("chi2_top_k", 25)), len(score_df))
top_rows = score_df.head(top_k).copy()

# Optional PRAD pathway refinement only if there is meaningful overlap
pathway_genes = [
    g.strip() for g in GENE_SET_PATH.read_text(encoding="utf-8").splitlines() if g.strip()
]

overlap = [g for g in top_rows["display_gene"].tolist() if g in pathway_genes]

if len(overlap) >= max(3, top_k // 5):
    refined_rows = top_rows[top_rows["display_gene"].isin(overlap)].copy()
    print("Pathway refinement applied.")
else:
    refined_rows = top_rows.copy()
    print("Pathway refinement skipped; using generic top-ranked features.")

pd.DataFrame(
    {
        "selected_gene": refined_rows["gene"].tolist(),
        "display_gene": refined_rows["display_gene"].tolist(),
    }
).to_csv(PROC_DIR / "selected_genes.csv", index=False)

score_df.to_csv(PROC_DIR / "chi2_scores.csv", index=False)

print("Feature selection complete")
print("Top genes (display):", top_rows["display_gene"].tolist())
print("Refined genes (display):", refined_rows["display_gene"].tolist())