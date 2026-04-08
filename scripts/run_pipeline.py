import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.mlops import (
    log_artifact,
    log_config_artifact,
    log_json_metrics,
    log_params_from_dict,
    log_pipeline_outputs,
    log_saved_models_as_mlflow_models,
    start_run,
)
from src.utils import load_yaml, resolve_config_path

DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo_dataset"
GENE_SET_DIR = DATA_DIR / "gene_sets"
GENE_SET_FILE = GENE_SET_DIR / "prad_pathway_genes.txt"
PROC_DIR = DATA_DIR / "processed"
TEMP_LOG_PATH = PROJECT_ROOT / ".pipeline.log.tmp"
TEMP_COMMANDS_PATH = PROJECT_ROOT / ".pipeline_commands.tmp"

DEMO_DIR.mkdir(parents=True, exist_ok=True)
GENE_SET_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

if not GENE_SET_FILE.exists():
    GENE_SET_FILE.write_text(
        "AR\nTP53\nBRCA1\nBRCA2\nPTEN\nMYC\nMKI67\nAKT1\nFOXA1\nERG\n",
        encoding="utf-8",
    )

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sim", "tcga"], default="sim")
parser.add_argument("--download-mode", choices=["api", "client"], default="api")
parser.add_argument("--clean", action="store_true")
parser.add_argument("--model-backend", choices=["pytorch"], default=None)
parser.add_argument(
    "--sim-signal-mode",
    choices=["linear", "interaction", "mixed"],
    default=None,
)
args = parser.parse_args()

config_path = resolve_config_path("configs/config.yaml")
cfg = load_yaml(str(config_path)) or {}

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

commands.extend(
    [
        "python scripts/preprocess_data.py",
        "python scripts/feature_selection.py",
        "python scripts/train_federated.py"
        + (f" --model-backend {args.model_backend}" if args.model_backend else ""),
        "python scripts/plot_predictions.py",
    ]
)

TEMP_COMMANDS_PATH.write_text("\n".join(commands) + "\n", encoding="utf-8")


def run_command(cmd: str, env: dict, log_handle) -> None:
    print("Running:", cmd)
    log_handle.write(f"\n$ {cmd}\n")
    log_handle.flush()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        shell=True,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout, end="")
        log_handle.write(result.stdout)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
        log_handle.write("\n--- STDERR ---\n")
        log_handle.write(result.stderr)
    log_handle.flush()

    if result.returncode != 0:
        raise RuntimeError(f"Failed at {cmd}")


run_name = (
    f"{args.mode}_{args.sim_signal_mode or 'default'}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)

try:
    with start_run(
        run_name=run_name,
        tags={
            "entrypoint": "run_pipeline",
            "mode": args.mode,
            "sim_signal_mode": args.sim_signal_mode or "default",
            "model_backend": args.model_backend or "pytorch",
            "config_path": str(config_path),
        },
    ):
        log_params_from_dict(
            {
                "mode": args.mode,
                "download_mode": args.download_mode,
                "clean": args.clean,
                "model_backend": args.model_backend or "pytorch",
                "sim_signal_mode": args.sim_signal_mode or "default",
                "config_path": str(config_path),
            },
            prefix="pipeline",
        )
        log_params_from_dict(cfg, prefix="config")
        log_config_artifact(str(config_path))

        child_env = os.environ.copy()
        child_env["FEDOMICS_MLFLOW_MANAGED_BY_PIPELINE"] = "1"

        with open(TEMP_LOG_PATH, "w", encoding="utf-8") as log_handle:
            for cmd in commands:
                run_command(cmd, child_env, log_handle)

        PROC_DIR.mkdir(parents=True, exist_ok=True)
        final_log_path = PROC_DIR / "pipeline.log"
        final_commands_path = PROC_DIR / "pipeline_commands.txt"
        final_log_path.write_text(
            TEMP_LOG_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        final_commands_path.write_text(
            TEMP_COMMANDS_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        log_json_metrics(PROC_DIR / "metrics.json", prefix="metrics")
        log_pipeline_outputs(PROC_DIR)
        log_saved_models_as_mlflow_models(PROC_DIR)

    print("Pipeline execution complete")
except Exception as exc:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_LOG_PATH.exists():
        final_log_path = PROC_DIR / "pipeline.log"
        final_log_path.write_text(
            TEMP_LOG_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        log_artifact(final_log_path, artifact_path="processed")
    sys.exit(str(exc))
finally:
    if TEMP_LOG_PATH.exists():
        TEMP_LOG_PATH.unlink()
    if TEMP_COMMANDS_PATH.exists():
        TEMP_COMMANDS_PATH.unlink()