import numpy as np
import cv2
from ..constants import CANON_PAD_PRIMARY, CANON_PAD_SECONDARY
from .features import trim_content_rgba

def phash_from_rgba(rgba, alpha_thr=1):
    return phash_from_canon_rgba(rgba, alpha_thr)

def _hamming64(a, b):
    return int((int(a) ^ int(b)).bit_count())

def canonical_gray_from_rgba(rgba: np.ndarray, alpha_thr: int = 1,
                             out_size: int = 32, pad_ratio: float = 0.08) -> np.ndarray:
    im = trim_content_rgba(rgba, alpha_thr)
    a = im[..., 3].astype(np.float32) / 255.0
    g = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = (g * a).astype(np.uint8)
    h, w = g.shape
    L = max(h, w)
    pad = int(L * pad_ratio)
    top = (L - h) // 2 + pad; bottom = L - h - (L - h)//2 + pad
    left = (L - w) // 2 + pad; right  = L - w - (L - w)//2 + pad
    gpad = cv2.copyMakeBorder(g, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    return cv2.resize(gpad, (out_size, out_size), interpolation=cv2.INTER_AREA)

def phash_from_canon_rgba(rgba: np.ndarray, alpha_thr: int = 1, pad_ratio: float = 0.08) -> int:
    gray = canonical_gray_from_rgba(rgba, alpha_thr, out_size=32, pad_ratio=pad_ratio)
    dct = cv2.dct(gray.astype(np.float32))[:8, :8]
    med = np.median(dct)
    bits = (dct > med).astype(np.uint8).flatten().tolist()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h

def phash_from_canon_alpha(rgba: np.ndarray, alpha_thr: int = 1, pad_ratio: float = 0.04) -> int:
    im = trim_content_rgba(rgba, alpha_thr)
    a = (im[..., 3] > alpha_thr).astype(np.uint8) * 255
    h, w = a.shape
    L = max(h, w)
    pad = int(L * pad_ratio)
    top = (L - h) // 2 + pad; bottom = L - h - (L - h)//2 + pad
    left = (L - w) // 2 + pad; right  = L - w - (L - w)//2 + pad
    apad = cv2.copyMakeBorder(a, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    apad = cv2.resize(apad, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(apad.astype(np.float32))[:8, :8]
    med = np.median(dct)
    bits = (dct > med).astype(np.uint8).flatten().tolist()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h

def phash_from_canon_uv(rgba: np.ndarray, alpha_thr: int = 1, pad_ratio: float = 0.08):
    im = trim_content_rgba(rgba, alpha_thr)
    a = im[..., 3].astype(np.float32) / 255.0
    yuv = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2YUV).astype(np.float32)
    U = (yuv[..., 1] * a).astype(np.uint8)
    V = (yuv[..., 2] * a).astype(np.uint8)

    def _canon(ch: np.ndarray) -> np.ndarray:
        h, w = ch.shape
        L = max(h, w); pad = int(L * pad_ratio)
        top = (L - h)//2 + pad; bottom = L - h - (L - h)//2 + pad
        left = (L - w)//2 + pad; right  = L - w - (L - w)//2 + pad
        ch = cv2.copyMakeBorder(ch, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        return cv2.resize(ch, (32, 32), interpolation=cv2.INTER_AREA)

    def _h64(img32: np.ndarray) -> int:
        dct = cv2.dct(img32.astype(np.float32))[:8, :8]
        med = np.median(dct)
        bits = (dct > med).astype(np.uint8).flatten().tolist()
        h = 0
        for b in bits: h = (h << 1) | int(b)
        return h

    return _h64(_canon(U)), _h64(_canon(V))

def phash_from_canon_edge(rgba: np.ndarray, alpha_thr: int = 1, pad_ratio: float = 0.08) -> int:
    g32 = canonical_gray_from_rgba(rgba, alpha_thr, out_size=32, pad_ratio=pad_ratio).astype(np.float32)
    gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    dct = cv2.dct(mag)[:8, :8]
    med = np.median(dct)
    bits = (dct > med).astype(np.uint8).flatten().tolist()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h

def rotate_rgba_expand(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    c = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    new_w = int(h*sin + w*cos + 0.5)
    new_h = int(h*cos + w*sin + 0.5)
    M[0,2] += (new_w/2.0) - c[0]
    M[1,2] += (new_h/2.0) - c[1]
    return cv2.warpAffine(
        rgba, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )

def best_rot_hamming_fast(hashA: int, rgbaB: np.ndarray, alpha_thr: int = 1, early_stop_at: int | None = None):
    best = 10**9
    best_ang = 0.0
    def score_at(angle: float) -> int:
        nonlocal best, best_ang
        R = rotate_rgba_expand(rgbaB, angle)
        hB = phash_from_canon_rgba(R, alpha_thr)
        d = int((hashA ^ hB).bit_count())
        if d < best:
            best, best_ang = d, float(angle)
        return d
    for a in (0, 90, 180, 270):
        if score_at(a) == 0:
            return 0, float(a)
    for step, radius in ((30, 60), (5, 15), (1, 3)):
        if early_stop_at is not None and best <= early_stop_at:
            break
        improved = True
        while improved:
            improved = False
            for a in (best_ang - step, best_ang + step):
                a = (a + 360.0) % 360.0
                d = score_at(a)
                if d == 0:
                    return 0, a
                if d < best:
                    improved = True
            if not improved and radius > step:
                lo = int((best_ang - radius + 360) % 360)
                hi = int((best_ang + radius) % 360)
                stepi = max(1, int(step))
                sweep = list(range(lo, 360, stepi)) + list(range(0, hi + 1, stepi)) if lo > hi else list(range(lo, hi + 1, stepi))
                for a in sweep:
                    d = score_at(a)
                    if d == 0:
                        return 0, float(a)
                    if d < best:
                        improved = True
    return best, best_ang