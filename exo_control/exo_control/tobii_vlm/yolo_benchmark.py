#!/usr/bin/env python3
"""
Benchmark YOLO inference time and throughput using the same model as tobii_vlm.

Usage examples:
  python3 -m exo_control.tobii_vlm.yolo_benchmark --images /path/to/val/images
  python3 -m exo_control.tobii_vlm.yolo_benchmark --video /path/to/video.mp4 --stride 2
"""

import argparse
import os
import statistics
import time
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


YOLO_MODEL_PATH = "/home/alessandro/exo_v2_ws/src/Exoskeleton/exo_control/yolo_weights/best-tobii-3objs-v2.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_image_paths(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(os.path.join(folder, name))
    return sorted(files)


def _read_images(paths: Iterable[str], flip_vertical: bool, max_frames: Optional[int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        if flip_vertical:
            frame = np.flip(frame, 0)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def _read_video_frames(
    video_path: str, flip_vertical: bool, stride: int, max_frames: Optional[int]
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            if flip_vertical:
                frame = np.flip(frame, 0)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def _sync_if_cuda(device: Optional[str]) -> None:
    if not torch.cuda.is_available():
        return
    if device is not None and device.lower().startswith("cpu"):
        return
    torch.cuda.synchronize()


def _benchmark(
    model: YOLO,
    frames: List[np.ndarray],
    conf: float,
    imgsz: Optional[int],
    device: Optional[str],
    warmup: int,
) -> List[float]:
    if not frames:
        return []

    warmup_frame = frames[0]
    for _ in range(max(0, warmup)):
        _sync_if_cuda(device)
        model.predict(warmup_frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        _sync_if_cuda(device)

    times_ms: List[float] = []
    for frame in frames:
        _sync_if_cuda(device)
        start = time.perf_counter()
        model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        _sync_if_cuda(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
    return times_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO inference time and throughput (tobii_vlm weights)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Folder containing images to benchmark.")
    group.add_argument("--video", help="Video file to benchmark.")
    parser.add_argument("--weights", default=YOLO_MODEL_PATH, help="Path to YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device override (e.g., cpu, 0, cuda:0).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--stride", type=int, default=1, help="Video frame stride.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames.")
    parser.add_argument("--flip-vertical", action="store_true", help="Flip frames vertically.")
    args = parser.parse_args()

    if args.images:
        if not os.path.isdir(args.images):
            print(f"[ERROR] Not a directory: {args.images}")
            return 2
        paths = _iter_image_paths(args.images)
        if not paths:
            print(f"[ERROR] No images found in {args.images}")
            return 2
        frames = _read_images(paths, args.flip_vertical, args.max_frames)
    else:
        if not os.path.isfile(args.video):
            print(f"[ERROR] Not a file: {args.video}")
            return 2
        frames = _read_video_frames(args.video, args.flip_vertical, args.stride, args.max_frames)

    if not frames:
        print("[ERROR] No frames loaded.")
        return 2

    print(f"[INFO] Loading YOLO weights: {args.weights}")
    model = YOLO(args.weights)
    print("[INFO] Using CUDA" if torch.cuda.is_available() else "[INFO] Using CPU")

    times_ms = _benchmark(model, frames, args.conf, args.imgsz, args.device, args.warmup)
    if not times_ms:
        print("[ERROR] No timing data collected.")
        return 2

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    total_s = sum(times_ms) / 1000.0
    fps = (len(times_ms) / total_s) if total_s > 0 else 0.0

    print(f"[RESULT] Frames: {len(times_ms)}")
    print(f"[RESULT] Inference time: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"[RESULT] Throughput: {fps:.2f} FPS")
    print(f"[LATEX] Inference time (ms/frame): {mean_ms:.2f}$\\pm${std_ms:.2f}")
    print(f"[LATEX] Throughput (FPS): {fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
