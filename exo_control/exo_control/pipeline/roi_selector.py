    from typing import Tuple, Optional
from PIL import Image


BBox = Tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, image_size) -> Optional[BBox]:
    """
    Clamp bbox to image boundary.

    image_size: (width, height)
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_roi(image: Image.Image, bbox: BBox, padding: int = 0) -> Optional[Image.Image]:
    """
    Crop ROI from a PIL image with optional padding.
    Returns a PIL.Image or None if bbox is invalid.
    """
    x1, y1, x2, y2 = bbox
    padded_bbox = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)

    valid_bbox = clamp_bbox(padded_bbox, image.size)
    if valid_bbox is None:
        return None

    x1, y1, x2, y2 = valid_bbox
    roi = image.crop((x1, y1, x2, y2))

    if roi.size[0] <= 0 or roi.size[1] <= 0:
        return None

    return roi
