import os, shutil
import fiftyone as fo
from pathlib import Path
from src.config import load_config

cfg = load_config(Path("src/config.yaml"))
ds = fo.load_dataset(cfg.dataset_name)

# fresh export
shutil.rmtree(cfg.exports_dir, ignore_errors=True)
cfg.exports_dir.mkdir(parents=True, exist_ok=True)

export_dir = os.fspath(cfg.exports_dir)

for split in ["train", "val", "test"]:
    view = ds.match_tags(split)
    print(split, "samples:", view.count("filepath"))

    view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections4",
        classes=cfg.classes,
        split=split,
    )
