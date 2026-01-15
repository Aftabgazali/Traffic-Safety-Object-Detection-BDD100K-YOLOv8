import fiftyone as fo
from pathlib import Path
from src.config import load_config

cfg = load_config(Path("src/config.yaml"))
ds = fo.load_dataset(cfg.dataset_name)

# open app and inspect detections4 on a few samples
session = fo.launch_app(ds)
session.wait()