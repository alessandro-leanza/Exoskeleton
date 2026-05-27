# Benchmark Guide — Yanchen

This document explains what was added on top of your pipeline code, what inputs are required, how to run the benchmark tests, and what you get as output.

The real-time test (Tobii glasses + ROS2 live) is handled by the supervisor. Your job is the offline benchmark described here.

---

## What was modified from your original code

### `exo_control/exo_control/pipeline/perception_pipeline.py`

| Change | Detail |
|--------|--------|
| CSV now logs VLM predictions | Added columns: `image_path`, `object_name`, `weight_level`, `fragility_level`, `json_ok`, `vlm_ms` |
| CSV path is configurable | Constructor now accepts `csv_path` parameter so each run writes its own file |

Before, the CSV only logged latency and trigger counts. VLM predictions (the actual weight/fragility output) were only printed to console and lost.

### `exo_control/exo_control/test_pipeline.py`

This file was rewritten. New features:

| Feature | Detail |
|---------|--------|
| **Video input** | Set `VIDEO_PATH` to a `.mp4`/`.avi` file. Frames are sampled at `SAMPLE_HZ` (default 3 Hz, matching the real-time YOLO loop rate) |
| **Image folder input** | Set `VIDEO_PATH = None` and point `IMAGE_DIR` to your Roboflow test split |
| **Ground truth matching** | `GROUND_TRUTH` dict maps object names to expected weight/fragility. VLM output is matched by keyword (case-insensitive, handles synonyms) |
| **Evaluation CSV** | Per-frame CSV comparing VLM predictions vs ground truth |
| **Auto-named output files** | Every run creates new files: `results/pipeline_log_{source}_{mode}_{timestamp}.csv` — no overwriting, no manual cleanup |
| **YOLO latency** | `yolo_ms` now measured and logged alongside VLM and pipeline latency |
| **Run summary** | Accuracy %, trigger rate %, mean latencies printed at the end of each run |

---

## Before you run

### 1. Set your paths

Open `exo_control/exo_control/test_pipeline.py` and edit the CONFIG section at the top:

```python
VIDEO_PATH   = None                          # set to "/path/to/video.mp4" for video mode
IMAGE_DIR    = "/path/to/roboflow/test/images"
YOLO_WEIGHTS = "/path/to/your/model.pt"
```

### 2. Extend the ground truth

The `GROUND_TRUTH` dict currently has 3 objects defined by the supervisor (screwdriver, drill, box). Add your objects following the same format:

```python
GROUND_TRUTH = {
    "screwdriver": {
        "keywords":  ["screwdriver", "cacciavite", "flathead", "phillips"],
        "weight":    "LOW",
        "fragility": "LOW",
    },
    "drill": {
        "keywords":  ["drill", "trapano", "power drill", "electric drill"],
        "weight":    "MEDIUM",
        "fragility": "MEDIUM",
    },
    "box": {
        "keywords":  ["box", "scatola", "cardboard", "carton", "package"],
        "weight":    "HEAVY",
        "fragility": "LOW",
    },
    # --- add your objects here ---
    "your_object": {
        "keywords":  ["name1", "name2", "synonym"],   # lowercase, what the VLM might say
        "weight":    "LOW",    # LOW / MEDIUM / HEAVY
        "fragility": "LOW",    # LOW / MEDIUM / HIGH
    },
}
```

**Keyword matching:** if the VLM returns `"large cardboard box"`, it matches `"box"` because `"box"` appears in the string. Add all synonyms and partial names you expect the VLM to use.

If the VLM returns a name that matches nothing, the row is logged as `NO_MATCH` and counted in the summary — use this to discover missing keywords.

---

## How to run

All outputs go to `results/` (created automatically). **You never need to delete files between runs.**

### Run 1 — Accuracy benchmark, bbox-trigger mode

```python
# in test_pipeline.py:
VIDEO_PATH          = None
IMAGE_DIR           = "/path/to/roboflow/test/images"
PIPELINE_MODE       = "bbox"
FORCE_STABLE_FRAMES = 0        # correct for static images
```

```bash
cd exo_control/exo_control
python test_pipeline.py
```

### Run 2 — Accuracy benchmark, grasp-episode mode

```python
PIPELINE_MODE = "grasp_episode"
# everything else unchanged
```

```bash
python test_pipeline.py
```

### Run 3 — Trigger rate benchmark, bbox-trigger mode

```python
VIDEO_PATH    = "/path/to/video.mp4"   # scene-camera video (no gaze needed)
PIPELINE_MODE = "bbox"
# FORCE_STABLE_FRAMES: leave at 0, video mode sets it to 2 automatically
```

```bash
python test_pipeline.py
```

### Run 4 — Trigger rate benchmark, grasp-episode mode

```python
PIPELINE_MODE = "grasp_episode"
# everything else unchanged
```

```bash
python test_pipeline.py
```

---

## Inputs

| Mode | Required input | Notes |
|------|---------------|-------|
| Image mode | Folder of `.jpg`/`.png` images | Use Roboflow test split (`test/images/`) |
| Video mode | `.mp4` or `.avi` file | Scene-camera footage only — no gaze data needed |
| Both modes | YOLO weights `.pt` file | Your trained model |
| Both modes | Qwen2.5-VL-7B-Instruct | Downloaded automatically from HuggingFace on first run |

**Video requirements for meaningful trigger rate results:**
- Record several complete grasp–hold–release cycles per object
- Include a pause between grabs (object not visible) so the grasp-episode logic resets
- 3–4 cycles per object is sufficient (~3–4 minutes total)
- No need for the Tobii glasses specifically — any camera works

---

## Outputs

Every run creates two files in `results/`:

```
results/
├── pipeline_log_images_bbox_20250527_143022.csv
├── evaluation_log_images_bbox_20250527_143022.csv
├── pipeline_log_images_grasp-episode_20250527_144501.csv
├── evaluation_log_images_grasp-episode_20250527_144501.csv
├── pipeline_log_video_bbox_20250527_150012.csv
├── evaluation_log_video_bbox_20250527_150012.csv
├── pipeline_log_video_grasp-episode_20250527_151233.csv
└── evaluation_log_video_grasp-episode_20250527_151233.csv
```

### `pipeline_log_*.csv` — one row per frame

| Column | Description |
|--------|-------------|
| `mode` | `bbox_trigger` or `grasp_episode` |
| `pipeline_latency_ms` | Total pipeline time (gate logic + VLM) |
| `vlm_call_count_this_run` | VLM calls in this frame (0 or 1) |
| `total_vlm_call_count` | Cumulative VLM calls |
| `triggered_this_run` | `True` if VLM fired |
| `skip_reason` | Why VLM was skipped (`not grasped`, `low confidence`, `unstable bounded box`, `same object`) |
| `num_objects` | Objects that passed all gates |
| `image_path` | Source image or video frame identifier |
| `object_name` | VLM free-text description |
| `weight_level` | VLM prediction: `LOW` / `MEDIUM` / `HEAVY` |
| `fragility_level` | VLM prediction: `LOW` / `MEDIUM` / `HIGH` |
| `json_ok` | `True` if VLM returned valid JSON |
| `vlm_ms` | VLM inference time (ms) |

### `evaluation_log_*.csv` — one row per frame

Extends pipeline_log with ground truth comparison:

| Column | Description |
|--------|-------------|
| `gt_object` | Matched canonical object name, or `NO_MATCH` |
| `gt_weight` | Ground truth weight level |
| `weight_match` | `True` / `False` |
| `gt_fragility` | Ground truth fragility level |
| `fragility_match` | `True` / `False` |
| `yolo_ms` | YOLO inference time (ms) |

### Console summary (printed at end of each run)

```
BENCHMARK SUMMARY
  Input mode:            IMAGES
  Pipeline mode:         bbox
  Total frames:          120
  VLM triggered:         87  (72.5%)
  Mean YOLO latency:     24.3 ms
  JSON parse failures:   2
  GT not matched:        3  (add keywords to GROUND_TRUTH)
  Weight accuracy:       74/82  (90.2%)
  Fragility accuracy:    61/82  (74.4%)
  Eval log:              results/evaluation_log_images_bbox_20250527_143022.csv
  Pipeline log:          results/pipeline_log_images_bbox_20250527_143022.csv
```

---

## What the 4 runs give you for the thesis

| Run | Input | Mode | Thesis metric |
|-----|-------|------|---------------|
| 1 | Roboflow images | bbox | Weight accuracy, fragility accuracy, VLM latency |
| 2 | Roboflow images | grasp_episode | Same metrics, different trigger strategy |
| 3 | Video | bbox | Trigger rate, YOLO+VLM latency in realistic temporal conditions |
| 4 | Video | grasp_episode | Same, for comparison with bbox |

Comparing runs 1 vs 2: which trigger strategy gives more VLM calls for the same image set?  
Comparing runs 3 vs 4: same question but on a realistic video sequence with stability gate active.  
Comparing runs 1 vs 2 on accuracy: does the trigger strategy affect which detections reach the VLM?

---

## Prompt and inference settings (fixed — do not change)

The VLM prompt is hardcoded in `pipeline/qwen_inference.py`. Do not modify it between runs — results must be comparable.

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Precision | FP16 |
| Decoding | Greedy (`do_sample=False`) |
| Max new tokens | 64 |
| YOLO confidence threshold | 0.30 (detection) + 0.70 (pipeline gate) |
