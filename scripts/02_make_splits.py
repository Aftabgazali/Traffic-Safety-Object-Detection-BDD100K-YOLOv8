import numpy as np
import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path
from src.config import load_config

cfg = load_config(Path("src/config.yaml"))
ds = fo.load_dataset(cfg.dataset_name)

# clear old tags
ds.untag_samples(["train", "val", "test"])

rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 42)

timevals = ds.distinct("timeofday.label")

for tv in timevals:
    view = ds.match(F("timeofday.label") == tv)
    ids = view.values("_id")
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    ds.select(ids[:n_train]).tag_samples("train")
    ds.select(ids[n_train:n_train+n_val]).tag_samples("val")
    ds.select(ids[n_train+n_val:]).tag_samples("test")

print("Counts:",
      "train", ds.match_tags("train").count("filepath"),
      "val", ds.match_tags("val").count("filepath"),
      "test", ds.match_tags("test").count("filepath"))
