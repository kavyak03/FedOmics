from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/demo_dataset")
OUT_DIR = Path("data/processed/qc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

center_files = sorted(RAW_DIR.glob("center_*_expression.csv"))
if not center_files:
    raise FileNotFoundError("No simulated center files found in data/demo_dataset")

dfs = []
for fp in center_files:
    df = pd.read_csv(fp)
    df["center"] = fp.stem
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Keep only numeric gene-like columns for QC
metadata_cols = {"label", "center", "signal_mode", "model_backend", "dataset_mode"}
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in metadata_cols and c != "label"]

if not feature_cols:
    raise ValueError("No numeric feature columns found for plotting.")

X = data[feature_cols].values.astype(float)
y = data["label"].values.astype(int)
centers = data["center"].values

# PCA
Xz = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xz)

plt.figure(figsize=(7, 5))
for label in sorted(np.unique(y)):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], alpha=0.7, label=f"label_{label}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of simulated samples by label")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_by_label.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
for center in sorted(np.unique(centers)):
    idx = centers == center
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], alpha=0.7, label=center)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of simulated samples by center")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_by_center.png", dpi=150)
plt.close()

# Correlation heatmap on top variable genes
gene_vars = data[feature_cols].var(axis=0).sort_values(ascending=False)
top_genes = gene_vars.head(min(30, len(gene_vars))).index.tolist()
corr = data[top_genes].corr().values

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(top_genes)), top_genes, rotation=90, fontsize=6)
plt.yticks(range(len(top_genes)), top_genes, fontsize=6)
plt.title("Correlation heatmap of top variable simulated genes")
plt.tight_layout()
plt.savefig(OUT_DIR / "gene_correlation_heatmap.png", dpi=150)
plt.close()

print("Saved QC plots to", OUT_DIR)