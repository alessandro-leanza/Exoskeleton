#!/usr/bin/env python3
import argparse
import asyncio
import base64
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import aiohttp
import cv2
import numpy as np

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_TIMEOUT_S = 20.0
WEIGHT_ESTIMATE_MAX_OUTPUT_TOKENS = 80
OBJECT_ESTIMATE_MAX_OUTPUT_TOKENS = 60


@dataclass
class MetadataRow:
    true_weight_g: Optional[float]
    true_object: Optional[str]
    true_grasped: Optional[str]


def _encode_jpeg_base64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_output_text(resp_json: dict) -> str:
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"]
    for out in resp_json.get("output", []) or []:
        for content in out.get("content", []) or []:
            if content.get("type") in ("output_text", "text"):
                return content.get("text", "")
    return ""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return text


async def _call_openai(
    image_bgr: np.ndarray,
    prompt: str,
    max_output_tokens: int,
    timeout_s: float,
    api_key: str,
) -> Tuple[str, float]:
    image_b64 = await asyncio.to_thread(_encode_jpeg_base64, image_bgr)
    payload = {
        "model": OPENAI_MODEL,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"OpenAI error {resp.status}: {body}")
    elapsed_s = time.perf_counter() - start

    try:
        resp_json = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response: {e}") from e

    output_text = _extract_output_text(resp_json)
    return _strip_code_fences(output_text), elapsed_s


def _parse_weight_g(text: str) -> Optional[float]:
    match = re.search(r"weight_g\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _parse_object_grasped(text: str) -> Tuple[Optional[str], Optional[str]]:
    obj = None
    grasped = None
    match_obj = re.search(r"object\s*[:=]\s*([^\n;]+)", text, re.IGNORECASE)
    match_grasped = re.search(r"grasped\s*[:=]\s*([^\n;]+)", text, re.IGNORECASE)
    if match_obj:
        obj = match_obj.group(1).strip()
    if match_grasped:
        grasped = match_grasped.group(1).strip()
    return obj, grasped


def _load_metadata(path: Optional[str]) -> Dict[str, MetadataRow]:
    if not path:
        return {}
    rows: Dict[str, MetadataRow] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("image") or row.get("filename") or "").strip()
            if not name:
                continue
            true_weight = row.get("true_weight_g") or row.get("true_weight") or ""
            true_object = row.get("true_object") or row.get("object") or ""
            true_grasped = row.get("true_grasped") or row.get("grasped") or ""
            rows[name] = MetadataRow(
                true_weight_g=float(true_weight) if true_weight else None,
                true_object=true_object.strip() or None,
                true_grasped=true_grasped.strip() or None,
            )
    return rows


def _iter_images(folder: str, exts: Iterable[str]) -> List[str]:
    all_files = []
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in exts:
            all_files.append(os.path.join(folder, name))
    return sorted(all_files)


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return 2

    meta = _load_metadata(args.metadata)
    images = _iter_images(args.images, {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        print(f"No images found in {args.images}")
        return 1

    weight_prompt = (
        "Identify the object and estimate its weight in grams based on the image. "
        "In your reasoning, explicitly say whether you can recognize the exact model; "
        "if yes, use that to justify the weight. "
        "Provide a brief reasoning (1-2 sentences), then a final line exactly in the form: "
        "weight_g: <number>."
    )

    object_prompt = (
        "Identify the object being grasped (if any) and whether the object is currently grasped. "
        "Be concise. Output two lines exactly in the form:\n"
        "object: <label>\n"
        "grasped: <yes|no|unknown>"
    )

    out_rows = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
        if args.flip_vertical:
            frame = np.flip(frame, 0)

        weight_text = ""
        weight_g = None
        weight_ms = None
        try:
            weight_text, weight_s = await _call_openai(
                frame, weight_prompt, WEIGHT_ESTIMATE_MAX_OUTPUT_TOKENS, OPENAI_TIMEOUT_S, api_key
            )
            weight_g = _parse_weight_g(weight_text)
            weight_ms = int(round(weight_s * 1000))
        except Exception as e:
            weight_text = f"ERROR: {e}"

        object_text = ""
        object_label = None
        grasped_label = None
        object_ms = None
        if not args.no_object:
            try:
                object_text, object_s = await _call_openai(
                    frame, object_prompt, OBJECT_ESTIMATE_MAX_OUTPUT_TOKENS, OPENAI_TIMEOUT_S, api_key
                )
                object_label, grasped_label = _parse_object_grasped(object_text)
                object_ms = int(round(object_s * 1000))
            except Exception as e:
                object_text = f"ERROR: {e}"

        meta_row = meta.get(img_name)
        out_rows.append(
            {
                "image": img_name,
                "weight_est_g": f"{weight_g:.2f}" if weight_g is not None else "",
                "true_weight_g": (
                    f"{meta_row.true_weight_g:.2f}" if meta_row and meta_row.true_weight_g is not None else ""
                ),
                "weight_inference_ms": f"{weight_ms}" if weight_ms is not None else "",
                "object_est": object_label or "",
                "true_object": meta_row.true_object if meta_row else "",
                "grasped_est": grasped_label or "",
                "true_grasped": meta_row.true_grasped if meta_row else "",
                "object_inference_ms": f"{object_ms}" if object_ms is not None else "",
                "weight_raw": weight_text,
                "object_raw": object_text,
            }
        )

    fieldnames = [
        "image",
        "weight_est_g",
        "true_weight_g",
        "weight_inference_ms",
        "object_est",
        "true_object",
        "grasped_est",
        "true_grasped",
        "object_inference_ms",
        "weight_raw",
        "object_raw",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline VLM evaluation on a folder of images.")
    p.add_argument("--images", required=True, help="Folder with images.")
    p.add_argument("--metadata", help="CSV with columns: image,true_weight_g,true_object,true_grasped")
    p.add_argument("--output", default="vlm_eval.csv", help="Output CSV path.")
    p.add_argument("--flip-vertical", action="store_true", help="Flip images vertically before inference.")
    p.add_argument("--no-object", action="store_true", help="Skip object/grasped inference.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
