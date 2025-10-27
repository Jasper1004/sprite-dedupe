import numpy as np
import os, json
import cv2
from PIL import Image, ImageFile
from PyQt5 import QtGui
from ..constants import SINGLES_BUCKET_SHIFT

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _visual_bucket(obj: dict | None, shift: int = SINGLES_BUCKET_SHIFT) -> tuple | None:
    """
    視覺後備：以多個 pHash 通道右移若干位形成粗 key。
    優先「新鍵」：primary/secondary/u/v/alpha/edge；若都沒有，再退回舊鍵：phash/phash_rgba。
    """
    if not isinstance(obj, dict):
        return None
    src = obj.get("features", obj)

    keys_try_new = ["phash_primary", "phash_secondary", "phash_u", "phash_v", "phash_alpha", "phash_edge"]
    vals = []
    for k in keys_try_new:
        v = src.get(k)
        try:
            if v is None:
                continue
            vals.append(int(v) >> shift)
        except Exception:
            continue
    if vals:
        return tuple(vals)

    keys_try_old = ["phash", "phash_rgba"]
    vals_old = []
    for k in keys_try_old:
        v = src.get(k)
        try:
            if v is None:
                continue
            vals_old.append(int(v) >> shift)
        except Exception:
            continue
    return tuple(vals_old) if vals_old else None

def read_image_rgba(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return np.array(im)

def rgba_to_rgb_alpha(rgba: np.ndarray):
    return rgba[..., :3].copy(), rgba[..., 3].copy()

def trim_and_pad_rgba(crop_rgba: np.ndarray, pad: int = 0) -> np.ndarray:
    if crop_rgba.size == 0:
        return crop_rgba
    alpha = crop_rgba[..., 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0:
        return crop_rgba
    y0, y1 = max(0, ys.min()-pad), min(crop_rgba.shape[0], ys.max()+1+pad)
    x0, x1 = max(0, xs.min()-pad), min(crop_rgba.shape[1], xs.max()+1+pad)
    return crop_rgba[y0:y1, x0:x1, :]

def to_gray(rgba: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)

def qpixmap_from_rgba(rgba: np.ndarray, max_w=400, max_h=400) -> QtGui.QPixmap:
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8, copy=False)
    h, w = rgba.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0) if w > 0 and h > 0 else 1.0
    if scale < 1.0:
        im = cv2.resize(rgba, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        im = rgba
    im = np.ascontiguousarray(im)
    h2, w2 = im.shape[:2]
    bytes_per_line = w2 * 4
    qimg = QtGui.QImage(bytes(im), w2, h2, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    qimg = qimg.copy()
    return QtGui.QPixmap.fromImage(qimg)

def make_thumb_rgba(rgba: np.ndarray, max_w=160, max_h=160) -> QtGui.QIcon:
    pm = qpixmap_from_rgba(rgba, max_w=max_w, max_h=max_h)
    return QtGui.QIcon(pm)

def write_results(project_root, pairs, id2item, out_path=None):
    """
    寫出 .image_cache/results.json
    - 維持原本 schema：{"pairs": [{"left_id": ..., "right_id": ..., "score": ...}, ...]}
    - 增強：
        1) 產生「同一張母圖內」的兩兩相似配對（同一格子圖不配）
        2) 允許「同圖不同 sub_id」的配對
        3) 僅排除「同一張同一 sub_id」→ self-match
        4) 無向邊去重（A↔B 視為同一對）
    """

    if out_path is None:
        out_path = os.path.join(project_root or "", ".image_cache", "results.json")

    def _split_key(k: str):
        if "#sub_" in k:
            u, s = k.split("#sub_", 1)
            try:
                return u, int(s)
            except Exception:
                return u, s
        return k, None

    def _pair_key(a: str, b: str):
        return tuple(sorted([a, b]))

    def _as_dict(p):
        if isinstance(p, dict):
            return {"left_id": p.get("left_id"), "right_id": p.get("right_id"), "score": p.get("score", 1.0)}
        return {"left_id": getattr(p, "left_id"), "right_id": getattr(p, "right_id"), "score": getattr(p, "score", 1.0)}

    def _hamm(a: int, b: int) -> int:
        return (int(a) ^ int(b)).bit_count()

    norm_pairs = []
    for p in pairs or []:
        d = _as_dict(p)
        if not d["left_id"] or not d["right_id"]:
            continue
        norm_pairs.append(d)

    feat_dir = os.path.join(project_root or "", ".image_cache", "features")
    if os.path.isdir(feat_dir):
        for fn in os.listdir(feat_dir):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(feat_dir, fn), "r", encoding="utf-8") as f:
                    feat = json.load(f)
            except Exception:
                continue

            subs = feat.get("sub_images") or []
            if len(subs) < 2:
                continue

            buckets = {}
            has_any = False
            for i, s in enumerate(subs):
                sig = None
                if isinstance(s.get("signature"), dict):
                    sig = s["signature"].get("semantic") or s["signature"].get("label") or s["signature"].get("name")
                elif isinstance(s.get("signature"), str):
                    sig = s.get("signature")

                if not sig:
                    vb = _visual_bucket(s, shift=SINGLES_BUCKET_SHIFT)
                    if vb:
                        sig = ("VB",) + vb 

                if sig:
                    has_any = True
                    buckets.setdefault(sig, []).append(i)

            if not has_any:
                continue

            uuid_ = feat.get("uuid") or fn[:-5]
            for _, idxs in buckets.items():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        left_id  = f"{uuid_}#sub_{idxs[a]}"
                        right_id = f"{uuid_}#sub_{idxs[b]}"
                        norm_pairs.append({"left_id": left_id, "right_id": right_id, "score": 1.0})

    dedup = set()
    filtered = []
    for d in norm_pairs:
        la, lb = d["left_id"], d["right_id"]
        ua, sa = _split_key(la)
        ub, sb = _split_key(lb)
        if ua == ub and sa == sb:
            continue
        key = _pair_key(la, lb)
        if key in dedup:
            continue
        dedup.add(key)
        filtered.append(d)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pairs": filtered}, f, ensure_ascii=False, indent=2)
    return len(filtered)