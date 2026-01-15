# scripts/05_train_yolo.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from src.config import load_config


def find_dataset_yaml(exports_dir: Path) -> Path:
    """
    Find the YOLO dataset yaml inside the export directory.
    Tries common names used by exporters.
    """
    candidates = [
        exports_dir / "data.yaml",
        exports_dir / "data.yml",
        exports_dir / "dataset.yaml",
        exports_dir / "dataset.yml",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: pick the first *.yml/*.yaml in the root of exports_dir
    yamls = list(exports_dir.glob("*.yml")) + list(exports_dir.glob("*.yaml"))
    if yamls:
        return yamls[0]

    raise FileNotFoundError(
        f"No dataset YAML found in {exports_dir}. Expected one of: "
        f"{', '.join([c.name for c in candidates])}"
    )


def safe_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "src" / "config.yaml")

    exports_dir = cfg.exports_dir
    data_yaml = find_dataset_yaml(exports_dir)

    print(f"[train] Using dataset YAML: {data_yaml}")
    print(f"[train] Classes: {cfg.classes}")
    print(f"[train] Model: {cfg.train_model_2}")
    print(f"[train] imgsz={cfg.train_imgsz_2} batch={cfg.train_batch_2} epochs={cfg.train_epochs_2} device={cfg.train_device_2}")

    # Train baseline
    model = YOLO(cfg.train_model_2)
    model.train(
        data=str(data_yaml),
        epochs=cfg.train_epochs_2,
        imgsz=cfg.train_imgsz_2,
        batch=cfg.train_batch_2,
        device=cfg.train_device_2,
        workers=cfg.train_workers_2,
        seed=cfg.seed,
        patience=cfg.train_patience_2,
        project=str(cfg.runs_dir),
        name=cfg.train_run_name_2,
        exist_ok=True,  # keep one run folder name stable; set False if you prefer unique runs
    )

    # Locate best weights from this run
    save_dir = Path(model.trainer.save_dir)  # ultralytics sets this
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"[train] Run dir: {save_dir}")
    print(f"[train] Best weights: {best_pt if best_pt.exists() else '(missing)'}")
    print(f"[train] Last weights: {last_pt if last_pt.exists() else '(missing)'}")

    # Evaluate on HOLDOUT test using best.pt if available
    eval_model = YOLO(str(best_pt if best_pt.exists() else last_pt))
    eval_split = cfg.eval_split

    metrics_obj = eval_model.val(
        data=str(data_yaml),
        split=eval_split,
        imgsz=cfg.train_imgsz_2,
        batch=cfg.train_batch_2,
        device=cfg.train_device_2,
    )

    # Ultralytics exposes metrics in a few forms depending on version.
    results_dict = None
    if hasattr(metrics_obj, "results_dict"):
        results_dict = metrics_obj.results_dict
    elif hasattr(metrics_obj, "metrics") and hasattr(metrics_obj.metrics, "results_dict"):
        results_dict = metrics_obj.metrics.results_dict

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(save_dir),
        "data_yaml": str(data_yaml),
        "weights_used_for_test": str(best_pt if best_pt.exists() else last_pt),
        "classes": cfg.classes,
        "train": {
            "model": cfg.train_model_2,
            "epochs": cfg.train_epochs_2,
            "imgsz": cfg.train_imgsz_2,
            "batch": cfg.train_batch_2,
            "device": cfg.train_device_2,
            "workers": cfg.train_workers_2,
            "seed": cfg.seed,
            "patience": cfg.train_patience_2,
            "run_name": cfg.train_run_name_2,
        },
        "eval": {
            "split": eval_split,
            "results_dict": results_dict,
        },
    }

    out_path = cfg.artifacts_dir / "metrics" / f"{cfg.train_run_name_2}_{eval_split}.json"
    safe_json_dump(out_path, summary)
    print(f"[train] Wrote metrics summary: {out_path}")


if __name__ == "__main__":
    main()
