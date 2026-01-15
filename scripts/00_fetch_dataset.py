from pathlib import Path
import fiftyone as fo
from src.config import load_config


cfg = load_config("src/config.yaml")
# Folder that contains samples.json, data/, fields/, etc
ROOT = Path(__file__).resolve().parents[1]          # repo root (traffic-safety-bdd100k)
DATASET_DIR = ROOT / "bdd100k_10k"
NAME = cfg.dataset_name

# <-- your folder name                         # <-- pick ANY name you want

print("dataset_dir:", DATASET_DIR)
print("exists:", DATASET_DIR.exists())

# If dataset name exists but is empty, delete it (DB only, not your files)
if NAME in fo.list_datasets():
    print(f"Deleting: {NAME}")
    fo.delete_dataset(NAME)

# Load existing non-empty dataset, else import from disk
if NAME in fo.list_datasets():
    ds = fo.load_dataset(NAME)
else:
    ds = fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.FiftyOneDataset,
        name=NAME,
        persistent=True,
    )

print("Loaded:", ds.name, "len:", len(ds))
print("Fields:", list(ds.get_field_schema().keys()))

fo.launch_app(ds).wait()
