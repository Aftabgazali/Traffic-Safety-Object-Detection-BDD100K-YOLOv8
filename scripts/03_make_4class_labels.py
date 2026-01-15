"""
Step 03: Create 4-class detections field: detections4
Inputs: dataset in FiftyOne DB, field 'detections'
Outputs: field 'detections4'
Safe to re-run: yes
"""

from pathlib import Path
import fiftyone as fo
from src.config import load_config

cfg = load_config(Path("src/config.yaml"))
ds = fo.load_dataset(cfg.dataset_name)

TOTAL = ds.count("filepath")
print("Total samples (via filepath count):", TOTAL)
print("Before:", ds.distinct("detections.detections.label"))

if "detections4" not in ds.get_field_schema():
    ds.add_sample_field(
        "detections4",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Detections,
    )

ids = []
vals = []

for i, s in enumerate(ds.select_fields(["detections"]).iter_samples(progress=False), start=1):
    dets = []
    if s.detections is not None:
        dets = s.detections.detections or []

    new_dets = []
    for d in dets:
        if d.label in cfg.person_merge_from:
            d2 = d.copy()
            d2.label = "person"
            new_dets.append(d2)
        elif d.label in cfg.keep:
            new_dets.append(d.copy())

    ids.append(s.id)
    vals.append(fo.Detections(detections=new_dets))

    if i % 500 == 0:
        print(f"Processed {i}/{TOTAL}")

# ✅ compatible bulk write
ds.select(ids).set_values("detections4", vals)

print("After:", ds.distinct("detections4.detections.label"))
print("Non-empty detections4:", ds.count("detections4"))