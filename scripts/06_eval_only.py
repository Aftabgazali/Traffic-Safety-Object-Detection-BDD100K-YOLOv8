from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from src.config import load_config


def find_dataset_yaml(exports_dir: Path) -> Path:
    for name in ["dataset.yml", "dataset.yaml", "data.yml", "data.yaml"]:
        p = exports_dir / name
        if p.exists():
            return p
    yamls = list(exports_dir.glob("*.yml")) + list(exports_dir.glob("*.yaml"))
    if yamls:
        return yamls[0]
    raise FileNotFoundError(f"No dataset yaml found in {exports_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "src" / "config.yaml")

    exports_dir = cfg.exports_dir
    data_yaml = find_dataset_yaml(exports_dir)

    run_dir = Path(cfg.runs_dir) / cfg.train_run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    weights = best_pt if best_pt.exists() else last_pt

    print(f"[eval-only] data: {data_yaml}", flush=True)
    print(f"[eval-only] weights: {weights}", flush=True)

    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {best_pt} and {last_pt}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=getattr(cfg, "eval_split", "test"),
        imgsz=getattr(cfg, "train_imgsz", 640),
        batch=getattr(cfg, "train_batch", 8),
        device=getattr(cfg, "train_device", "0"),
        project=str(getattr(cfg, "runs_dir", repo_root / "runs")),
        name=f"{cfg.train_run_name}_eval_test",
        exist_ok=True,
    )

    results_dict = getattr(metrics, "results_dict", None)
    artifacts_dir = getattr(cfg, "artifacts_dir", repo_root / "artifacts")
    out = Path(artifacts_dir) / "metrics" / f"{cfg.train_run_name}_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "data_yaml": str(data_yaml),
                "weights": str(weights),
                "results_dict": results_dict,
            },
            f,
            indent=2,
        )

    print(f"[eval-only] wrote: {out}", flush=True)


if __name__ == "__main__":
    main()