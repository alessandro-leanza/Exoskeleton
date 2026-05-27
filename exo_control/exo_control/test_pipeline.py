import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from ultralytics import YOLO

from pipeline.qwen_inference import Qwen2_5VLBackend
from pipeline.perception_pipeline import PerceptionPipeline


# ================= CONFIG =================
IMAGE_DIR    = "/workspace/test_images"   # folder with test images
YOLO_WEIGHTS = "/workspace/models/yolov8/best.pt"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

YOLO_CONF          = 0.30
PIPELINE_MODE      = "bbox"   # "bbox" or "grasp_episode"
FORCE_STABLE_FRAMES = 0       # 0 = disable stability gate for offline testing
ONLY_USE_GRASPED   = False    # False = keep both classes; True = grasped only
# ==========================================


# ================= GROUND TRUTH ===========
# Keywords are matched case-insensitively against the VLM's object_name output.
# Add more objects here or ask Yanchen to extend the list.
GROUND_TRUTH = {
    "screwdriver": {
        "keywords":  ["screwdriver", "cacciavite", "flathead", "phillips", "torx", "flatblade"],
        "weight":    "LOW",       # ~100 g
        "fragility": "LOW",       # metal tool, robust
    },
    "drill": {
        "keywords":  ["drill", "trapano", "power drill", "electric drill", "hand drill", "cordless drill"],
        "weight":    "MEDIUM",    # ~1 kg
        "fragility": "MEDIUM",    # electronics inside
    },
    "box": {
        "keywords":  ["box", "scatola", "cardboard", "carton", "crate", "package", "parcel", "container"],
        "weight":    "HEAVY",     # ~5 kg
        "fragility": "LOW",       # cardboard deforms but does not break
    },
}
EVAL_LOG_PATH = "evaluation_log.csv"
# ==========================================


def match_ground_truth(object_name: str) -> Optional[Dict[str, Any]]:
    """Return GT entry whose keywords appear in object_name (case-insensitive), or None."""
    name_lower = object_name.lower()
    for canonical, data in GROUND_TRUTH.items():
        if any(kw in name_lower for kw in data["keywords"]):
            return {"object": canonical, **data}
    return None


def init_eval_csv(path: str) -> csv.writer:
    file_exists = os.path.isfile(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "image_path",
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
            "pipeline_latency_ms",
            "skip_reason",
        ])
    return f, writer


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted([p for p in folder_path.iterdir() if p.is_file() and is_image_file(p)])


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


def main() -> None:
    image_paths = collect_images(IMAGE_DIR)
    print(f"Found {len(image_paths)} image(s)")

    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading Qwen backend...")
    backend  = Qwen2_5VLBackend(QWEN_MODEL_ID)
    pipeline = PerceptionPipeline(backend)
    pipeline.stable_frames_required = FORCE_STABLE_FRAMES

    eval_file, eval_writer = init_eval_csv(EVAL_LOG_PATH)

    # running counters for summary
    total = triggered = weight_ok = fragility_ok = no_gt_match = json_fail = 0

    try:
        for idx, img_path in enumerate(image_paths, start=1):
            print("\n" + "=" * 90)
            print(f"[{idx}/{len(image_paths)}] {img_path.name}")

            frame = Image.open(img_path).convert("RGB")

            yolo_results = yolo.predict(source=str(img_path), conf=YOLO_CONF, verbose=False)
            detections   = yolo_to_pipeline_detections(yolo_results[0], only_use_grasped=ONLY_USE_GRASPED)

            print(f"  YOLO: {len(detections)} detection(s)")
            for i, det in enumerate(detections, 1):
                print(f"    {i}. class={det['class']}  conf={det['confidence']:.3f}  bbox={det['bbox']}")

            result = pipeline.run(frame, detections, mode=PIPELINE_MODE, image_path=str(img_path))

            triggered_now = result["triggered_this_run"]
            latency       = result["pipeline_latency_ms"]
            skip          = result["skip_reason"]
            total        += 1
            triggered    += int(triggered_now)

            print(f"  Pipeline: triggered={triggered_now}  latency={latency:.1f} ms  skip={skip}")

            if result["results"]:
                for obj in result["results"]:
                    vlm_name  = obj["object_name"]
                    vlm_w     = obj["weight_level"]
                    vlm_f     = obj["fragility_level"]
                    jok       = obj["json_ok"]
                    vms       = obj["vlm_ms"]

                    if not jok:
                        json_fail += 1

                    gt = match_ground_truth(vlm_name)

                    if gt:
                        w_match = vlm_w == gt["weight"]
                        f_match = vlm_f == gt["fragility"]
                        weight_ok    += int(w_match)
                        fragility_ok += int(f_match)
                        w_sym = "OK" if w_match else "WRONG"
                        f_sym = "OK" if f_match else "WRONG"
                        print(f"  VLM  → object='{vlm_name}'  weight={vlm_w}  fragility={vlm_f}  json={jok}  vlm_ms={vms:.0f}")
                        print(f"  GT   → object='{gt['object']}'  weight={gt['weight']}  fragility={gt['fragility']}")
                        print(f"  Eval → weight={w_sym}  fragility={f_sym}")
                        eval_writer.writerow([
                            str(img_path), triggered_now, jok,
                            vlm_name, gt["object"],
                            vlm_w, gt["weight"], w_match,
                            vlm_f, gt["fragility"], f_match,
                            f"{vms:.1f}", f"{latency:.1f}", skip,
                        ])
                    else:
                        no_gt_match += 1
                        print(f"  VLM  → object='{vlm_name}'  weight={vlm_w}  fragility={vlm_f}  json={jok}")
                        print(f"  GT   → NO MATCH — add '{vlm_name}' keywords to GROUND_TRUTH if needed")
                        eval_writer.writerow([
                            str(img_path), triggered_now, jok,
                            vlm_name, "NO_MATCH",
                            vlm_w, "", "",
                            vlm_f, "", "",
                            f"{vms:.1f}", f"{latency:.1f}", skip,
                        ])
            else:
                print("  No VLM result (not triggered or no grasped object).")
                eval_writer.writerow([
                    str(img_path), False, "",
                    "", "", "", "", "", "", "", "",
                    "", f"{latency:.1f}", skip,
                ])

            eval_file.flush()

    finally:
        pipeline.close()
        eval_file.close()

        # ---- final summary ----
        n_triggered = triggered
        n_with_gt   = n_triggered - no_gt_match - json_fail
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY")
        print(f"  Total images:          {total}")
        print(f"  VLM triggered:         {n_triggered}  ({100*n_triggered/total:.1f}%)")
        print(f"  JSON parse failures:   {json_fail}")
        print(f"  GT matches found:      {n_triggered - no_gt_match}")
        print(f"  GT not matched:        {no_gt_match}  (add keywords to GROUND_TRUTH)")
        if n_with_gt > 0:
            print(f"  Weight accuracy:       {weight_ok}/{n_with_gt}  ({100*weight_ok/n_with_gt:.1f}%)")
            print(f"  Fragility accuracy:    {fragility_ok}/{n_with_gt}  ({100*fragility_ok/n_with_gt:.1f}%)")
        print(f"  Eval log:              {EVAL_LOG_PATH}")
        print(f"  Pipeline log:          pipeline_log.csv")


if __name__ == "__main__":
    main()
