#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import sys

# ====== CONFIG DI DEFAULT ======
YOLO_MODEL_PATH = '/home/alessandro/exo_v2_ws/src/Exoskeleton/exo_control/yolo_weights/best-realsense-tobii.pt'
DEFAULT_INPUT_BASENAME  = '/home/alessandro/Videos/icra/video_glasses_icra_task.mp4'
DEFAULT_OUTPUT_BASENAME = '/home/alessandro/Videos/icra/video_glasses_icra_NO.mp4'

# ====== NUOVE OPZIONI ======
# Se vuoi forzare sempre una label personalizzata, mettila qui.
# Esempi:
#   OVERRIDE_LABEL = "Box-Not Grasped"  -> stampa sempre questa label
#   OVERRIDE_LABEL = None               -> usa le label del modello
OVERRIDE_LABEL = "Box-Not Grasped"

# Considera solo queste classi come "box" (filtra gli altri oggetti)
TARGET_CLASS_NAMES = {"Box-Grasped", "Box-Not Grasped"}

# soglie per-classe (come nel live)
MIN_CONF = {
    "Box-Grasped": 0.6,
    "Box-Not Grasped": 0.6,
}

# conf globale minima passata a YOLO (tienila <= min(MIN_CONF.values()))
GLOBAL_MIN_CONF = 0.30


def _resolve_local_path(basename_or_path: str, default_exts=('.mp4', '.avi', '.mkv', '.webm')) -> str:
    """
    Se basename_or_path è un path esistente, lo usa.
    Altrimenti, se non ha estensione, prova ad aggiungere in ordine .mp4/.avi/.mkv
    cercando nella cartella dello script.
    """
    if os.path.exists(basename_or_path):
        return basename_or_path

    root, ext = os.path.splitext(basename_or_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if ext == '':
        for e in default_exts:
            cand = os.path.join(script_dir, root + e)
            if os.path.exists(cand):
                return cand
        cand = os.path.join(script_dir, basename_or_path)
        return cand
    else:
        cand = os.path.join(script_dir, basename_or_path)
        return cand


def annotate_video(input_path: str, output_path: str, weights_path: str, flip_vert: bool = False):
    print(f"[YOLO] Loading: {weights_path}")
    model = YOLO(weights_path)
    print("[YOLO] Using CUDA" if torch.cuda.is_available() else "[YOLO] Using CPU")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire input video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Impossibile aprire output video: {output_path}")

    print(f"[INFO] Writing {output_path} ({width}x{height} @ {fps:.2f} fps)")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if flip_vert:
            frame = cv2.flip(frame, 0)

        # Inferenza (conf globale bassa; filtro per-classe dopo)
        results = model.predict(frame, conf=GLOBAL_MIN_CONF, verbose=False)
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                try:
                    cls_id = int(b.cls[0])
                    p = float(b.conf[0])
                    raw_label = model.names.get(cls_id, str(cls_id))

                    # filtra: consideriamo solo le classi di "box"
                    if TARGET_CLASS_NAMES and raw_label not in TARGET_CLASS_NAMES:
                        continue

                    # filtro per-classe
                    min_req = MIN_CONF.get(raw_label, GLOBAL_MIN_CONF)
                    if p < min_req:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width - 1, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # label: usa override se definito, altrimenti label del modello
                    label_text = OVERRIDE_LABEL if OVERRIDE_LABEL is not None else raw_label
                    # se vuoi aggiungere la conf: f"{label_text} {p:.2f}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    pad = 4
                    patch_w = tw + 2*pad
                    patch_h = th + baseline + 2*pad

                    top_y = y1 - 4 - patch_h
                    left_x = x1
                    if top_y < 0:
                        top_y = y1 + 2

                    patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
                    cv2.putText(patch, label_text, (pad, patch_h - baseline - pad),
                                font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

                    # incolla patch nei limiti
                    x0 = max(0, left_x); y0 = max(0, top_y)
                    x_end = min(width, left_x + patch_w); y_end = min(height, top_y + patch_h)
                    if x_end > x0 and y_end > y0:
                        px0 = max(0, -left_x); py0 = max(0, -top_y)
                        px_end = px0 + (x_end - x0); py_end = py0 + (y_end - y0)
                        frame[y0:y_end, x0:x_end] = patch[py0:py_end, px0:px_end]

                except Exception as e:
                    print(f"[WARN] annotazione frame {frame_idx}: {e}")

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("[OK] Finito.")


def main():
    parser = argparse.ArgumentParser(description="Annota un video con YOLO (bbox+label) offline.")
    parser.add_argument("--input",  help="Percorso video input. Default: ./video_in_0[.mp4/.avi/.mkv]")
    parser.add_argument("--output", help="Percorso video output. Default: ./video_out_0.mp4")
    parser.add_argument("--weights", help=f"Percorso pesi YOLO. Default: {YOLO_MODEL_PATH}")
    parser.add_argument("--flip-vert", action="store_true", help="Flip verticale prima di annotare")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # risolvi input
    in_arg = args.input if args.input else DEFAULT_INPUT_BASENAME
    input_path = _resolve_local_path(in_arg)
    if not os.path.exists(input_path):
        print(f"[ERRORE] Input non trovato: {input_path}", file=sys.stderr)
        sys.exit(1)

    # risolvi output (sempre nella cartella dello script)
    out_arg = args.output if args.output else DEFAULT_OUTPUT_BASENAME + ".mp4"
    output_path = out_arg if os.path.isabs(out_arg) else os.path.join(script_dir, out_arg)

    # pesi
    weights_path = args.weights if args.weights else YOLO_MODEL_PATH

    annotate_video(input_path, output_path, weights_path, flip_vert=args.flip_vert)


if __name__ == "__main__":
    main()
