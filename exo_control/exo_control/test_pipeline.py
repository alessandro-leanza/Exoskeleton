import os
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from ultralytics import YOLO

from pipeline.qwen_inference import Qwen2_5VLBackend
from pipeline.perception_pipeline import PerceptionPipeline


# ================= CONFIG =================
IMAGE_DIR = "/workspace/test_images"          # change it to image folder
YOLO_WEIGHTS = "/workspace/models/yolov8/best.pt"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

YOLO_CONF = 0.30
PIPELINE_MODE = "bbox"          # 
FORCE_STABLE_FRAMES = 0         #  set to 0 to get rid of the stable frame gate in perception pipeline
ONLY_USE_GRASPED = False        # False = 保留两类；True = 只保留 grasped
# ==========================================


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted([p for p in folder_path.iterdir() if p.is_file() and is_image_file(p)])


def map_yolo_label(raw_label: str) -> str | None:
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
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        raw_label = names.get(cls_id, str(cls_id))
        mapped = map_yolo_label(raw_label)

        if mapped is None:
            continue
        if only_use_grasped and mapped != "grasped":
            continue

        xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        detections.append(
            {
                "bbox": tuple(xyxy),
                "class": mapped,
                "confidence": conf,
            }
        )

    return detections


def main() -> None:
    image_paths = collect_images(IMAGE_DIR)
    print(f"Found {len(image_paths)} image(s)")

    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading Qwen backend...")
    backend = Qwen2_5VLBackend(QWEN_MODEL_ID)
    pipeline = PerceptionPipeline(backend)

    # close the stable frame test in offline test
    pipeline.stable_frames_required = FORCE_STABLE_FRAMES

    try:
        for idx, img_path in enumerate(image_paths, start=1):
            print("\n" + "=" * 90)
            print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")

            frame = Image.open(img_path).convert("RGB")

            yolo_results = yolo.predict(
                source=str(img_path),
                conf=YOLO_CONF,
                verbose=False
            )
            result0 = yolo_results[0]

            detections = yolo_to_pipeline_detections(
                result0,
                only_use_grasped=ONLY_USE_GRASPED,
            )

            print(f"YOLO detections: {len(detections)}")
            for i, det in enumerate(detections, start=1):
                print(
                    f"  {i}. class={det['class']}, "
                    f"conf={det['confidence']:.3f}, bbox={det['bbox']}"
                )

            pipeline_result = pipeline.run(
                frame,
                detections,
                mode=PIPELINE_MODE,
            )

            print("Pipeline summary:")
            print(f"  mode: {pipeline_result['mode']}")
            print(f"  latency_ms: {pipeline_result['pipeline_latency_ms']:.2f}")
            print(f"  vlm_call_count_this_run: {pipeline_result['vlm_call_count_this_run']}")
            print(f"  total_vlm_call_count: {pipeline_result['total_vlm_call_count']}")
            print(f"  triggered_this_run: {pipeline_result['triggered_this_run']}")
            print(f"  skip_reason: {pipeline_result['skip_reason']}")
            print(f"  num_objects: {pipeline_result['num_objects']}")

            if pipeline_result["results"]:
                for j, obj in enumerate(pipeline_result["results"], start=1):
                    print(f"  Object {j}:")
                    print(f"    object_name: {obj['object_name']}")
                    print(f"    weight_level: {obj['weight_level']}")
                    print(f"    fragility_level: {obj['fragility_level']}")
                    print(f"    json_ok: {obj['json_ok']}")
                    print(f"    vlm_ms: {obj['vlm_ms']:.2f}")
                    print(f"    raw_output: {obj['raw_output']}")
            else:
                print("  No VLM result for this image.")

    finally:
        pipeline.close()
        print("\nClosed pipeline CSV file.")


if __name__ == "__main__":
    main()
