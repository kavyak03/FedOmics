import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo_dataset"
GENE_SET_DIR = DATA_DIR / "gene_sets"
GENE_SET_FILE = GENE_SET_DIR / "prad_pathway_genes.txt"

DEMO_DIR.mkdir(parents=True, exist_ok=True)
GENE_SET_DIR.mkdir(parents=True, exist_ok=True)
if not GENE_SET_FILE.exists():
    GENE_SET_FILE.write_text(
        "AR\nTP53\nBRCA1\nBRCA2\nPTEN\nMYC\nMKI67\nAKT1\nFOXA1\nERG\n",
        encoding="utf-8",
    )

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sim", "tcga"], default="sim")
parser.add_argument("--download-mode", choices=["api", "client"], default="api")
parser.add_argument("--clean", action="store_true")
parser.add_argument("--model-backend", choices=["pytorch", "tensorflow"], default=None)
parser.add_argument("--sim-signal-mode", choices=["linear", "interaction", "mixed"], default=None)
args = parser.parse_args()

commands = []
if args.clean:
    commands.append("python scripts/clean_pipeline.py")

if args.mode == "sim":
    gen_cmd = "python scripts/generate_sim_data.py"
    if args.sim_signal_mode:
        gen_cmd += f" --signal-mode {args.sim_signal_mode}"
    commands.append(gen_cmd)
    commands.append("python scripts/plot_sim_qc.py")
elif args.mode == "tcga":
    commands.append(f"python scripts/download_tcga.py --download-mode {args.download_mode}")

commands.extend([
    "python scripts/preprocess_data.py",
    "python scripts/feature_selection.py",
    "python scripts/train_federated.py" + (f" --model-backend {args.model_backend}" if args.model_backend else ""),
    "python scripts/plot_predictions.py",
])

for cmd in commands:
    print("Running:", cmd)
    rc = os.system(cmd)
    if rc != 0:
        sys.exit(f"Failed at {cmd}")

print("Pipeline execution complete")
