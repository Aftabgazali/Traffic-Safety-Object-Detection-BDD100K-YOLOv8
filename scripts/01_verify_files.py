from pathlib import Path
import fiftyone as fo
from src.config import load_config

cfg = load_config("src/config.yaml")

name = cfg.dataset_name  # <-- don't auto-add _local here
print("Loading dataset:", name)
ds = fo.load_dataset(name)

print("Dataset object:", ds)
print("Dataset name:", ds.name)
print("Persistent:", ds.persistent)

# Counts (these should NOT be 0 if dataset has samples)
print("len(ds):", len(ds))
print("ds.count():", ds.count())
print("ds.count('filepath'):", ds.count("filepath"))

# Grab a few samples and show filepaths exist
first3 = ds.take(3)
print("First 3 filepaths:")
for s in first3:
    print("  ", s.filepath, "| exists:", Path(s.filepath).is_file())

# Now do the missing scan (but also track how many we actually iterated)
missing = 0
seen = 0
for s in ds.select_fields(["filepath"]).iter_samples(progress=True):
    seen += 1
    if not Path(s.filepath).is_file():
        missing += 1

print("Iterated:", seen)
print("Missing:", missing, "out of", seen)
