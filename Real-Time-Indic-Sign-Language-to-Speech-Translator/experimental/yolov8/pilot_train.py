"""Run one resource-test epoch and validation on the prepared YOLO dataset."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_CONFIG = EXPERIMENT_DIR / "pilot_data.yaml"
CHECKPOINT = EXPERIMENT_DIR / "weights" / "yolov8n.pt"
RUNS_DIR = EXPERIMENT_DIR / "runs"
RUN_NAME = f"pilot_retry_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
RUN_DIR = RUNS_DIR / RUN_NAME
SETTINGS = {
    "data": str(DATASET_CONFIG),
    "model": str(CHECKPOINT),
    "epochs": 1,
    "imgsz": 320,
    "batch": 1,
    "device": "cpu",
    "workers": 0,
    "cache": False,
    "seed": 42,
    "val": True,
    "test_split_used": False,
}


def checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"Prepared dataset config not found: {DATASET_CONFIG}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Verified checkpoint not found: {CHECKPOINT}")
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {RUN_DIR}")

    RUN_DIR.mkdir(parents=True)
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
        "checkpoint_sha256": checkpoint_sha256(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "status": "running",
    }
    metadata_path = RUN_DIR / "pilot_metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    started = time.monotonic()
    try:
        dataset = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
        dataset_root = Path(dataset["path"])
        if not dataset_root.is_absolute():
            dataset["path"] = str((EXPERIMENT_DIR / dataset_root).resolve())
        model = YOLO(str(CHECKPOINT))
        if model.task != "detect":
            raise RuntimeError(f"Expected detection checkpoint, got {model.task!r}")
        results = model.train(
            data=dataset,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            epochs=1,
            imgsz=320,
            batch=1,
            device="cpu",
            workers=0,
            cache=False,
            seed=42,
            val=True,
            plots=False,
            save=True,
        )
        metric_values = getattr(results, "results_dict", {})
        record["validation_metrics"] = {
            str(key): float(value)
            for key, value in metric_values.items()
            if isinstance(value, (int, float))
        }
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
