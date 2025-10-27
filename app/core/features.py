import numpy as np
import cv2
from ..constants import SHAPE_ALPHA_THR
from ..utils.image_io import trim_and_pad_rgba

def trim_content_rgba(rgba: np.ndarray, alpha_thr: int = 1) -> np.ndarray:
    a = rgba[..., 3]
    ys, xs = np.where(a > alpha_thr)
    if xs.size == 0:
        return rgba
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return rgba[y0:y1, x0:x1, :]

def content_area_ratio(rgba: np.ndarray, alpha_thr: int = 1) -> float:
    im = trim_content_rgba(rgba, alpha_thr)
    a = im[..., 3] >= alpha_thr
    return float(a.mean())

def gray_hist32(rgba: np.ndarray, alpha_thr: int = 1) -> np.ndarray:
    a = rgba[..., 3] >= alpha_thr
    if not a.any():
        return np.zeros(32, dtype=np.float32)
    y = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
    y = y[a]
    hist, _ = np.histogram(y, bins=32, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    return hist / max(s, 1.0)

def chisq_dist(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    s = p + q + 1e-9
    return 0.5 * float((d * d / s).sum())

def crop_aspect_ratio(rgba: np.ndarray, alpha_thr: int = 1) -> float:
    im = trim_content_rgba(rgba, alpha_thr); h, w = im.shape[:2]
    return (w / max(1, h))