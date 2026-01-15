# Traffic Safety Object Detection (BDD100K → YOLOv8)

End-to-end object detection pipeline for traffic safety on a BDD100K (HF) subset, focused on **4 classes**:
**car**, **person**, **rider**, **traffic light**. Built with **FiftyOne** for dataset handling and **Ultralytics YOLOv8** for training/evaluation.

**North-star goals**
- Raise **person recall** (miss fewer people)
- Raise **traffic light recall** + improve **box tightness** (small/far objects)

---

## Results (holdout test)

### Baseline: YOLOv8n @ 640 (30 epochs)
| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.592 | 0.447 | 0.470 | 0.232 |
| car | 0.690 | 0.674 | 0.711 | 0.445 |
| person | 0.605 | 0.445 | 0.476 | 0.216 |
| rider | 0.474 | 0.249 | 0.259 | 0.106 |
| traffic light | 0.597 | 0.420 | 0.434 | 0.159 |

### Final: YOLOv8s @ 1024 (50 epochs)
| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 0.697 | 0.574 | 0.625 | 0.328 |
| car | 0.768 | 0.761 | 0.817 | 0.529 |
| person | 0.729 | 0.545 | 0.640 | 0.324 |
| rider | 0.596 | 0.379 | 0.406 | 0.210 |
| traffic light | 0.694 | 0.609 | 0.637 | 0.250 |

**Key improvements (test)**
- **Person recall:** 0.445 → **0.545** (+0.100)
- **Traffic light recall:** 0.420 → **0.609** (+0.189)
- **Traffic light mAP50-95 (tightness):** 0.159 → **0.250** (+0.091)
- **Overall mAP50-95:** 0.232 → **0.328** (+0.096)

> Note: `rider` has very few examples (58 instances in test), so its metrics are more volatile.

---

## Repo structure

```

traffic-safety-bdd100k/
artifacts/
class_map/            # label remap metadata (json)
splits/               # split file for reproducibility (json)
metrics/              # metrics summaries (json)
bdd100k_10k/            # local FiftyOne dataset state (not for git)
exports/
yolo_bdd4/            # YOLO-format dataset export (not for git)
runs/                   # Ultralytics training/eval outputs (not for git)
scripts/                # step-by-step pipeline
src/
config.yaml           # single source of truth (paths + params)
config.py             # config loader

````

---

## Setup

### 1) Create environment
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# or Git Bash
source .venv/Scripts/activate
````

### 2) Install dependencies

```bash
pip install -U pip
pip install ultralytics fiftyone datasets pyyaml tqdm opencv-python
```

---

## Configuration (`src/config.yaml`)

The pipeline reads everything from a single config file:

* dataset name (FiftyOne): `bdd100k_hf_10k_local`
* export dir: `exports/yolo_bdd4`
* class list: `["car", "person", "rider", "traffic light"]`
* label mapping: `pedestrian` + `other person` → `person`
* split policy: seed=42, ratios=80/10/10, stratified by `timeofday`
* training presets:

  * `train_yolov8n` (baseline)
  * `train_yolov8s` (final)
* evaluation split: `test`

---

## Pipeline (reproduce end-to-end)

Run from repo root.

### Step 00 — Fetch dataset

```bash
python -m scripts.00_fetch_dataset
```

### Step 01 — Verify files

```bash
python -m scripts.01_verify_files
```

### Step 02 — Create splits (timeofday-stratified)

Writes `artifacts/splits/split_seed42.json` and tags samples in FiftyOne.

```bash
python -m scripts.02_make_splits
```

### Step 03 — Create 4-class labels (`detections4`)

Adds a new label field `detections4` that:

* keeps: `car`, `rider`, `traffic light`
* maps: `pedestrian` + `other person` → `person`
* drops: all other classes

```bash
python -m scripts.03_make_4class_labels
```

### Step 04 — Export YOLO dataset

Exports into `exports/yolo_bdd4/` (train/val/test images + labels + dataset.yaml).

```bash
python -m scripts.04_export_yolo
```

### Step 05 — Train and evaluate

**Baseline (YOLOv8n @ 640):**

```bash
python -m scripts.05_train_yolo --preset train_yolov8n
```

**Final (YOLOv8s @ 1024):**

```bash
python -m scripts.05_train_yolo --preset train_yolov8s
```

**Eval-only (test):**

```bash
yolo val model="runs/yolov8s_1024_bdd4_seed42_final/weights/best.pt" data="exports/yolo_bdd4/dataset.yaml" split=test imgsz=1024 batch=1 device=0
```

> If your `05_train_yolo.py` does not currently accept `--preset`, either add it or just run the final preset directly from config. The repo already stores both presets in `src/config.yaml`.

---

## Why these choices?

### Why stratify by `timeofday`?

We want to analyze robustness across lighting conditions fairly. Stratifying by `timeofday` ensures train/val/test have similar proportions of day/night/dawn/dusk so:

* the holdout test is not accidentally “harder”
* improvements aren’t just caused by a biased split

### Why resolution + bigger model for the final run?

Small/far objects (traffic lights) suffer when they occupy very few pixels. Increasing resolution gives the model more signal per object and generally improves:

* **recall** (miss fewer small objects)
* **localization** (tighter boxes → higher mAP50-95)

A slightly larger model (`yolov8s`) adds capacity to learn tougher patterns (night glare, partial occlusions), complementing the resolution increase.

---

## Outputs you should look at

After training, check:

* `runs/<run_name>/weights/best.pt` — final model
* `runs/<run_name>/results.png` — training curves
* `runs/detect/val*` — sample predictions
* `artifacts/metrics/*.json` — saved metrics summary (easy to paste into README)

---

## Next: real-time inference demo (sellable)

Planned demo (simple + effective):

* Load `best.pt`
* Run inference on a short driving clip
* Render boxes + labels + confidence
* Keep confidence around ~0.2–0.3 to balance precision/recall

We’ll add a `scripts/06_infer_video.py` and a short `demo/` folder once the repo is pushed.

---


## Notes on dataset & licensing

This project uses BDD100K data via HuggingFace + local export. Please check the dataset’s official license/terms before redistributing images/labels.
(Links, if you want them handy:)

```text
HuggingFace dataset:
https://huggingface.co/datasets/dgural/bdd100k

BDD100K official:
https://bdd-data.berkeley.edu/
```
---

## Author
Aftab (UdeM) — Traffic-safety perception baseline + improvement run (YOLOv8)