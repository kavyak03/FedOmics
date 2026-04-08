import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

try:
    import mlflow
except Exception:
    mlflow = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mlflow_enabled() -> bool:
    return mlflow is not None and os.getenv("FEDOMICS_DISABLE_MLFLOW", "0") != "1"


def configure_mlflow(project_root: Path = PROJECT_ROOT) -> bool:
    if not mlflow_enabled():
        return False

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("FEDOMICS_MLFLOW_TRACKING_URI")
    )

    if not tracking_uri:
        db_path = (project_root / "mlflow.db").resolve()
        tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("FEDOMICS_MLFLOW_EXPERIMENT", "FedOmics")
    mlflow.set_experiment(experiment_name)
    return True


@contextmanager
def start_run(run_name: str = None, tags: Dict[str, Any] = None):
    active = configure_mlflow()
    if not active:
        yield False
        return

    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        if tags:
            clean_tags = {str(k): str(v) for k, v in tags.items() if v is not None}
            mlflow.set_tags(clean_tags)
        yield True


def sanitize_key(key: str) -> str:
    return (
        str(key)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def flatten_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_prefix = f"{prefix}.{sanitize_key(k)}" if prefix else sanitize_key(k)
            flat.update(flatten_dict(v, next_prefix))
    elif isinstance(obj, list):
        flat[prefix] = ",".join(str(v) for v in obj)
    else:
        flat[prefix] = obj
    return flat


def log_params_from_dict(params: Dict[str, Any], prefix: str = "") -> None:
    if not mlflow_enabled() or not params:
        return

    flat = flatten_dict(params, prefix=prefix)
    clean = {}
    for k, v in flat.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            clean[k] = v
        else:
            clean[k] = str(v)
    if clean:
        mlflow.log_params(clean)


def log_metrics_from_dict(metrics: Dict[str, Any], prefix: str = "") -> None:
    if not mlflow_enabled() or not metrics:
        return

    flat = flatten_dict(metrics, prefix=prefix)
    clean = {}
    for k, v in flat.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v is not None:
            clean[k] = float(v)
    if clean:
        mlflow.log_metrics(clean)


def log_json_metrics(path: Path, prefix: str = "") -> None:
    path = Path(path)
    if not path.exists() or not mlflow_enabled():
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    log_metrics_from_dict(payload, prefix=prefix)


def log_artifact(path: Path, artifact_path: str = None) -> None:
    path = Path(path)
    if not path.exists() or not mlflow_enabled():
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts_in_dir(path: Path, artifact_path: str = None) -> None:
    path = Path(path)
    if not path.exists() or not mlflow_enabled():
        return
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def log_pipeline_outputs(processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    if not processed_dir.exists() or not mlflow_enabled():
        return

    for name in [
        "metrics.json",
        "predictions.csv",
        "roc_points.csv",
        "threshold.json",
        "selected_genes.csv",
        "chi2_scores.csv",
        "pipeline.log",
        "pipeline_commands.txt",
        "selected_genes_cv.csv",
    ]:
        p = processed_dir / name
        if p.exists():
            log_artifact(p, artifact_path="processed")

    qc_dir = processed_dir / "qc"
    if qc_dir.exists():
        log_artifacts_in_dir(qc_dir, artifact_path="processed/qc")

    models_dir = processed_dir / "models"
    if models_dir.exists():
        log_artifacts_in_dir(models_dir, artifact_path="processed/models")


def log_config_artifact(config_path: str) -> None:
    path = Path(config_path)
    if path.exists():
        log_artifact(path, artifact_path="config")


def log_saved_models_as_mlflow_models(processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    models_dir = processed_dir / "models"
    if not models_dir.exists() or not mlflow_enabled():
        return

    federated_path = models_dir / "federated_pytorch_model.pt"
    feature_names_path = models_dir / "feature_names.json"

    if federated_path.exists():
        try:
            import torch
            import mlflow.pytorch
            from src.model import GeneExpressionNet

            input_dim = None
            if feature_names_path.exists():
                with open(feature_names_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                input_dim = len(payload.get("feature_names", []))

            if input_dim is not None and input_dim > 0:
                model = GeneExpressionNet(input_dim=input_dim)
                state_dict = torch.load(federated_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()

                example_input = torch.randn(1, input_dim)

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="federated_pytorch_model",
                    input_example=example_input,
                )
        except Exception as exc:
            print(f"Warning: failed to log saved PyTorch model to MLflow: {exc}")


def log_standalone_training_run(
    run_name: str,
    params: Dict[str, Any],
    config_path: str,
    processed_dir: Path,
) -> None:
    with start_run(
        run_name=run_name,
        tags={"entrypoint": "train_federated", "managed_by": "standalone"},
    ):
        log_params_from_dict(params)
        if Path(config_path).exists():
            try:
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                log_params_from_dict(cfg, prefix="config")
            except Exception:
                pass
            log_config_artifact(config_path)

        log_json_metrics(Path(processed_dir) / "metrics.json", prefix="metrics")
        log_pipeline_outputs(Path(processed_dir))
        log_saved_models_as_mlflow_models(Path(processed_dir))