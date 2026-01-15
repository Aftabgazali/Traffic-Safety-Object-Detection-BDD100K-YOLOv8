from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass(frozen=True)
class Config:
    dataset_name: str
    exports_dir: Path
    runs_dir: Path
    splits_file: Path
    class_map_file: Path
    artifacts_dir: Path

    source_field: str
    filtered_field: str
    classes: List[str]
    keep: List[str]
    person_merge_from: List[str]

    seed: int
    ratios: Dict[str, float]
    stratify_field: str

    export_dataset_type: str
    
    # train_yolov8n
    train_model: str
    train_epochs: int
    train_imgsz: int
    train_batch: int
    train_device: str
    train_workers: int
    train_patience: int
    train_run_name: str
    
    # train_yolov8s
    train_model_2: str
    train_epochs_2: int
    train_imgsz_2: int
    train_batch_2: int
    train_device_2: str
    train_workers_2: int
    train_patience_2: int
    train_run_name_2: str

    # eval
    eval_split: str


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    def p(key: str) -> Path:
        # resolve paths relative to repo root (config file's parent is /src)
        return (path.parent.parent / raw["paths"][key]).resolve()

    return Config(
        dataset_name=raw["project"]["dataset_name"],
        exports_dir=p("exports_dir"),
        artifacts_dir=p("artifacts_dir"),
        runs_dir=p("runs_dir"),
        splits_file=p("splits_file"),
        class_map_file=p("class_map_file"),
        source_field=raw["labels"]["source_field"],
        filtered_field=raw["labels"]["filtered_field"],
        classes=list(raw["labels"]["classes"]),
        keep=list(raw["labels"]["keep"]),
        person_merge_from=list(raw["labels"]["person_merge_from"]),
        seed=int(raw["split"]["seed"]),
        ratios=dict(raw["split"]["ratios"]),
        stratify_field=str(raw["split"]["stratify_field"]),
        export_dataset_type=str(raw["export"]["dataset_type"]),
        
        train_model=str(raw["train_yolov8n"]["model"]),
        train_epochs=int(raw["train_yolov8n"]["epochs"]),
        train_imgsz=int(raw["train_yolov8n"]["imgsz"]),
        train_batch=int(raw["train_yolov8n"]["batch"]),
        train_device=str(raw["train_yolov8n"]["device"]),
        train_workers=int(raw["train_yolov8n"]["workers"]),
        train_patience=int(raw["train_yolov8n"]["patience"]),
        train_run_name=str(raw["train_yolov8n"]["run_name"]),
        
        train_model_2=str(raw["train_yolov8s"]["model"]),
        train_epochs_2=int(raw["train_yolov8s"]["epochs"]),
        train_imgsz_2=int(raw["train_yolov8s"]["imgsz"]),
        train_batch_2=int(raw["train_yolov8s"]["batch"]),
        train_device_2=str(raw["train_yolov8s"]["device"]),
        train_workers_2=int(raw["train_yolov8s"]["workers"]),
        train_patience_2=int(raw["train_yolov8s"]["patience"]),
        train_run_name_2=str(raw["train_yolov8s"]["run_name"]),
        
        eval_split=str(raw["eval"]["split"]),
    )