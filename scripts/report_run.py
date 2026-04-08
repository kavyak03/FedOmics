from pathlib import Path
import json
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.utils import load_yaml

CFG = load_yaml("configs/config.yaml")
PROC = Path("data/processed")
OUT = Path("outputs/reports")
OUT.mkdir(parents=True, exist_ok=True)

metrics = json.loads((PROC / "metrics.json").read_text()) if (PROC / "metrics.json").exists() else {}
selected = pd.read_csv(PROC / "selected_genes.csv")["selected_gene"].tolist() if (PROC / "selected_genes.csv").exists() else []
center_counts = {}
for fp in sorted(PROC.glob("center_*_train.csv")) + sorted(PROC.glob("center_*_val.csv")):
    center_counts[fp.stem] = int(len(pd.read_csv(fp)))

manifest = {
    "model_backend": CFG.get("model_backend"),
    "learning_rate": CFG.get("learning_rate"),
    "epochs": CFG.get("epochs"),
    "num_centers": CFG.get("num_centers"),
    "selected_genes": selected,
    "center_counts": center_counts,
    "metrics": metrics,
}

(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
lines = [
    "# FedOmics Run Summary",
    "",
    f"- Model backend: `{manifest['model_backend']}`",
    f"- Epochs: `{manifest['epochs']}`",
    f"- Learning rate: `{manifest['learning_rate']}`",
    f"- Number of centers: `{manifest['num_centers']}`",
    "",
    "## Selected genes",
]
for g in selected:
    lines.append(f"- {g}")
lines.extend(["", "## Metrics"])
for group, vals in metrics.items():
    if isinstance(vals, dict):
        lines.append(f"### {group}")
        for center, center_vals in vals.items():
            if isinstance(center_vals, dict):
                lines.append(f"- **{center}**: accuracy={center_vals.get('accuracy')}, f1={center_vals.get('f1')}, auroc={center_vals.get('auroc')}")
(OUT / "run_summary.md").write_text("".join(lines), encoding="utf-8")
print("Run report written to outputs/reports/")
