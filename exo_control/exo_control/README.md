# Perception Pipeline (YOLO + ROI + VLM)

This project implements a perception pipeline for object understanding:

YOLO → ROI Cropping → Vision-Language Model (Qwen2.5-VL)

The system is designed for real-time use with Tobii glasses, while also supporting offline testing using image datasets.

---

## Project Structure

project_root/
│
├── pipeline/
│   ├── perception_pipeline.py
│   ├── qwen_inference.py
│   ├── roi_selector.py
│   └── output_parser.py
│
├── tobbi_vlm/
│   └── __main__.py
│
├── test_pipeline.py
├── README.md
└── pipeline_log.csv

---

## Requirements

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- Transformers
- CUDA (recommended)

---

## Running Modes

There are two different ways to run this project.

---

### 1. Real-Time Mode (Tobii Glasses)

Entry point:


python tobbi_vlm/__main__.py

Description:

This mode is designed for real-time perception using:

- Tobii eye-tracking glasses 
- Live camera stream 
- ROS2 communication 

This mode does not support offline image testing because it depends on:

- real-time frame input 
- gaze-trigger logic 
- ROS node execution 

Requirements:

- Tobii glasses connected 
- ROS2 environment properly set up 
- Camera stream available 

---

### 2. Offline Testing Mode (Recommended for Development)

To evaluate the perception pipeline on a dataset, use:

python test_pipeline.py

What This Script Does:

- Loads images from a folder 
- Runs YOLO detection 
- Converts detections to pipeline format 
- Runs the perception pipeline:
  - ROI cropping 
  - Qwen2.5-VL inference 
  - JSON parsing 
- Saves logs to:

pipeline_log.csv

---

## Configuration

Inside test_pipeline.py, modify:

IMAGE_DIR = "/path/to/your/images" 
YOLO_WEIGHTS = "/path/to/yolo/best.pt" 
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct" 

---

## Important Notes

1. Stability Constraint

The pipeline uses a stability constraint:

self.stable_frames_required = 2

For offline testing, this must be disabled, otherwise the VLM may never be triggered.

Add in test_pipeline.py:

pipeline.stable_frames_required = 0

2. "grasped" Class Requirement

The pipeline only runs VLM when:

class == "grasped"

If your dataset contains only "not_grasped":

- VLM will not be triggered 
- CSV will still be written, but results will be empty 

For offline testing, you can force all detections to be treated as grasped:

det["class"] = "grasped"

3. CSV Output

Each run appends results to:

pipeline_log.csv

Columns include:

- latency 
- VLM call count 
- trigger status 
- skip reason 
- number of detected objects 

---

## Recommended Testing Setup

PIPELINE_MODE = "bbox" 
pipeline.stable_frames_required = 0 
ONLY_USE_GRASPED = False 
