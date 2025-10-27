import numpy as np
import cv2
from typing import List, Tuple
from ..constants import SPRITESHEET_MIN_SEGMENTS_DEFAULT as _MIN_SEGS
from ..constants import SPRITESHEET_MIN_COVERAGE_DEFAULT as _MIN_COVER
from ..utils.image_io import rgba_to_rgb_alpha, trim_and_pad_rgba

def alpha_cc_boxes(rgba: np.ndarray, alpha_thr: int, min_area: int, min_size: int) -> List[Tuple[int,int,int,int]]:
    _, alpha = rgba_to_rgb_alpha(rgba)
    mask = (alpha > alpha_thr).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    boxes = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if w < min_size or h < min_size:
            continue
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    return boxes

def coverage_ratio(shape_hw, boxes) -> float:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    for (x,y,w_,h_) in boxes:
        m[y:y+h_, x:x+w_] = 1
    return float(m.sum()) / (h*w) if (h*w) else 0.0

def is_spritesheet(rgba, boxes, min_segments=_MIN_SEGS, min_cover=_MIN_COVER):
    if len(boxes) >= (min_segments * 2):
        return True
    if len(boxes) >= min_segments and coverage_ratio(rgba.shape[:2], boxes) >= min_cover:
        return True
    h, w = rgba.shape[:2]
    ar = (w / max(1, h)) if h > 0 else 0
    if len(boxes) >= 2 and (ar > 1.8 or ar < 0.55):
        return True
    return False

def crop_boxes(rgba: np.ndarray, boxes, pad: int = 0):
    outs = []
    for (x,y,w,h) in boxes:
        crop = rgba[y:y+h, x:x+w, :]
        crop = trim_and_pad_rgba(crop, pad=pad)
        outs.append(crop)
    return outs