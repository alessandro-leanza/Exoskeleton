import csv
import os
import time
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from pipeline.qwen_inference import Qwen2_5VLBackend
from pipeline.perception_pipeline import PerceptionPipeline


# ================= CONFIG =================
# --- Input: set VIDEO_PATH to use video mode, leave None to use IMAGE_DIR ---
VIDEO_PATH = None              # e.g. "/path/to/recording.mp4"  or  None
SAMPLE_HZ  = 3.0              # frames per second to sample from video (match real-time YOLO rate)

IMAGE_DIR    = "/workspace/test_images"
YOLO_WEIGHTS = "/workspace/models/yolov8/best.pt"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

YOLO_CONF    = 0.30
PIPELINE_MODE = "grasp_episode"  # "bbox" or "grasp_episode"

# Stability gate:
#   0  → disabled (correct for static images — no temporal continuity)
#   2  → enabled  (correct for video input — matches real-time behaviour)
FORCE_STABLE_FRAMES = 0

ONLY_USE_GRASPED = False
# ==========================================


# ================= GROUND TRUTH ===========
# Keywords matched case-insensitively against the VLM's object_name output.
# Yanchen can extend this dict with more objects.
GROUND_TRUTH = {
    "screwdriver": {
        "keywords":  ["screwdriver", "cacciavite", "flathead", "phillips", "torx", "flatblade"],
        "weight":    "LOW",      # ~100 g
        "fragility": "LOW",      # metal tool, robust
    },
    "drill": {
        "keywords":  ["drill", "trapano", "power drill", "electric drill", "hand drill", "cordless drill"],
        "weight":    "MEDIUM",   # ~1 kg
        "fragility": "MEDIUM",   # electronics inside
    },
    "box": {
        "keywords":  ["box", "scatola", "cardboard", "carton", "crate", "package", "parcel", "container"],
        "weight":    "HEAVY",    # ~5 kg
        "fragility": "LOW",      # cardboard deforms but does not break
    },
}
EVAL_LOG_PATH = "evaluation_log.csv"
# ==========================================


# ------------------------------------------------------------------
# Frame sources
# ------------------------------------------------------------------

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frames_from_images(folder: str) -> Generator[Tuple[Image.Image, str], None, None]:
    """Yield (PIL Image, label) for every image in folder, sorted by name."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = sorted(p for p in folder_path.iterdir() if p.is_file() and is_image_file(p))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    for p in paths:
        yield Image.open(p).convert("RGB"), str(p)


def frames_from_video(video_path: str, sample_hz: float) -> Generator[Tuple[Image.Image, str], None, None]:
    """
    Yield (PIL Image, label) sampled at sample_hz from a video file.
    Skips frames so that the pipeline sees exactly sample_hz frames/s of video time,
    matching the real-time YOLO loop rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, round(src_fps / sample_hz))
    expected = total_frames // frame_step

    name = Path(video_path).name
    print(f"  Video: {src_fps:.1f} fps  →  sampling every {frame_step} frame(s) @ {sample_hz} Hz")
    print(f"  Total source frames: {total_frames}  →  ~{expected} frames to process")

    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_s = frame_idx / src_fps
            label = f"{name}::frame_{frame_idx:05d}_t_{timestamp_s:.2f}s"
            yield Image.fromarray(rgb), label
        frame_idx += 1

    cap.release()


# ------------------------------------------------------------------
# Ground truth & evaluation helpers
# ------------------------------------------------------------------

def match_ground_truth(object_name: str) -> Optional[Dict[str, Any]]:
    name_lower = object_name.lower()
    for canonical, data in GROUND_TRUTH.items():
        if any(kw in name_lower for kw in data["keywords"]):
            return {"object": canonical, **data}
    return None


def init_eval_csv(path: str):
    file_exists = os.path.isfile(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "frame_id",
            "triggered",
            "json_ok",
            "vlm_object_name",
            "gt_object",
            "vlm_weight",
            "gt_weight",
            "weight_match",
            "vlm_fragility",
            "gt_fragility",
            "fragility_match",
            "vlm_ms",
            "yolo_ms",
            "pipeline_latency_ms",
            "skip_reason",
        ])
    return f, writer


# ------------------------------------------------------------------
# YOLO helpers
# ------------------------------------------------------------------

def map_yolo_label(raw_label: str) -> Optional[str]:
    if raw_label == "grasped":
        return "grasped"
    if raw_label == "not_grasped":
        return "not_grasped"
    return None


def yolo_to_pipeline_detections(result, only_use_grasped: bool = False) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections
    names = result.names
    for b in boxes:
        cls_id    = int(b.cls[0])
        conf      = float(b.conf[0])
        raw_label = names.get(cls_id, str(cls_id))
        mapped    = map_yolo_label(raw_label)
        if mapped is None:
            continue
        if only_use_grasped and mapped != "grasped":
            continue
        xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        detections.append({"bbox": tuple(xyxy), "class": mapped, "confidence": conf})
    return detections


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    # --- choose input source ---
    if VIDEO_PATH:
        source_label = f"VIDEO  {VIDEO_PATH}  @ {SAMPLE_HZ} Hz"
        frame_source = frames_from_video(VIDEO_PATH, SAMPLE_HZ)
        stable_frames = FORCE_STABLE_FRAMES if FORCE_STABLE_FRAMES != 0 else 2
        print(f"Video mode: FORCE_STABLE_FRAMES overridden to {stable_frames} "
              f"(set FORCE_STABLE_FRAMES explicitly to override)")
        if FORCE_STABLE_FRAMES != 0:
            stable_frames = FORCE_STABLE_FRAMES
    else:
        source_label = f"IMAGES  {IMAGE_DIR}"
        frame_source = frames_from_images(IMAGE_DIR)
        stable_frames = FORCE_STABLE_FRAMES

    print(f"Input:  {source_label}")
    print(f"Mode:   {PIPELINE_MODE}   stable_frames={stable_frames}")

    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading Qwen backend...")
    backend  = Qwen2_5VLBackend(QWEN_MODEL_ID)
    pipeline = PerceptionPipeline(backend)
    pipeline.stable_frames_required = stable_frames

    eval_file, eval_writer = init_eval_csv(EVAL_LOG_PATH)

    total = triggered = weight_ok = fragility_ok = no_gt_match = json_fail = 0
    total_yolo_ms = 0.0

    try:
        for idx, (frame, frame_id) in enumerate(frame_source, start=1):
            print("\n" + "=" * 90)
            print(f"[{idx}] {frame_id}")

            # YOLO — timed
            t0 = time.perf_counter()
            yolo_results = yolo.predict(source=np.array(frame), conf=YOLO_CONF, verbose=False)
            yolo_ms = (time.perf_counter() - t0) * 1000
            total_yolo_ms += yolo_ms

            detections = yolo_to_pipeline_detections(yolo_results[0], only_use_grasped=ONLY_USE_GRASPED)
            print(f"  YOLO: {len(detections)} detection(s)  ({yolo_ms:.0f} ms)")
            for i, det in enumerate(detections, 1):
                print(f"    {i}. class={det['class']}  conf={det['confidence']:.3f}  bbox={det['bbox']}")

            result = pipeline.run(frame, detections, mode=PIPELINE_MODE, image_path=frame_id)

            triggered_now = result["triggered_this_run"]
            latency       = result["pipeline_latency_ms"]
            skip          = result["skip_reason"]
            total        += 1
            triggered    += int(triggered_now)

            print(f"  Pipeline: triggered={triggered_now}  latency={latency:.1f} ms  skip={skip}")

            if result["results"]:
                for obj in result["results"]:
                    vlm_name = obj["object_name"]
                    vlm_w    = obj["weight_level"]
                    vlm_f    = obj["fragility_level"]
                    jok      = obj["json_ok"]
                    vms      = obj["vlm_ms"]

                    if not jok:
                        json_fail += 1

                    gt = match_ground_truth(vlm_name)

                    if gt:
                        w_match = vlm_w == gt["weight"]
                        f_match = vlm_f == gt["fragility"]
                        weight_ok    += int(w_match)
                        fragility_ok += int(f_match)
                        print(f"  VLM  → object='{vlm_name}'  weight={vlm_w}  fragility={vlm_f}  vlm_ms={vms:.0f}")
                        print(f"  GT   → object='{gt['object']}'  weight={gt['weight']}  fragility={gt['fragility']}")
                        print(f"  Eval → weight={'OK' if w_match else 'WRONG'}  fragility={'OK' if f_match else 'WRONG'}")
                        eval_writer.writerow([
                            frame_id, triggered_now, jok,
                            vlm_name, gt["object"],
                            vlm_w, gt["weight"], w_match,
                            vlm_f, gt["fragility"], f_match,
                            f"{vms:.1f}", f"{yolo_ms:.1f}", f"{latency:.1f}", skip,
                        ])
                    else:
                        no_gt_match += 1
                        print(f"  VLM  → object='{vlm_name}'  weight={vlm_w}  fragility={vlm_f}")
                        print(f"  GT   → NO MATCH — add '{vlm_name}' keywords to GROUND_TRUTH if needed")
                        eval_writer.writerow([
                            frame_id, triggered_now, jok,
                            vlm_name, "NO_MATCH",
                            vlm_w, "", "",
                            vlm_f, "", "",
                            f"{vms:.1f}", f"{yolo_ms:.1f}", f"{latency:.1f}", skip,
                        ])
            else:
                print("  No VLM result.")
                eval_writer.writerow([
                    frame_id, False, "",
                    "", "", "", "", "", "", "", "",
                    "", f"{yolo_ms:.1f}", f"{latency:.1f}", skip,
                ])

            eval_file.flush()

    finally:
        pipeline.close()
        eval_file.close()

        n_with_gt = triggered - no_gt_match - json_fail
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY")
        print(f"  Input mode:            {'VIDEO' if VIDEO_PATH else 'IMAGES'}")
        print(f"  Pipeline mode:         {PIPELINE_MODE}")
        print(f"  Total frames:          {total}")
        print(f"  VLM triggered:         {triggered}  ({100*triggered/max(total,1):.1f}%)")
        if total > 0:
            print(f"  Mean YOLO latency:     {total_yolo_ms/total:.1f} ms")
        print(f"  JSON parse failures:   {json_fail}")
        print(f"  GT not matched:        {no_gt_match}  (add keywords to GROUND_TRUTH)")
        if n_with_gt > 0:
            print(f"  Weight accuracy:       {weight_ok}/{n_with_gt}  ({100*weight_ok/n_with_gt:.1f}%)")
            print(f"  Fragility accuracy:    {fragility_ok}/{n_with_gt}  ({100*fragility_ok/n_with_gt:.1f}%)")
        print(f"  Eval log:              {EVAL_LOG_PATH}")
        print(f"  Pipeline log:          pipeline_log.csv")


if __name__ == "__main__":
    main()
