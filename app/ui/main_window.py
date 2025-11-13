# app/ui/main_window.py
import os, sys, shutil, tempfile, uuid
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from dataclasses import dataclass
from collections import defaultdict, OrderedDict


from ..constants import (
    LIGHT_QSS, DARK_QSS, VERSION,
    ROT_DEG_STEP_DEFAULT, INCLUDE_FLIP_TEST_DEFAULT,
    PHASH_HAMMING_MAX_DEFAULT, PHASH_HAMMING_MAX_INTRA_DEFAULT,
    ALPHA_THR_DEFAULT, MIN_AREA_DEFAULT, MIN_SIZE_DEFAULT,
    SPRITESHEET_MIN_SEGMENTS_DEFAULT, SPRITESHEET_MIN_COVERAGE_DEFAULT,
    SINGLES_BUCKET_SHIFT,
    PHASH_HAMMING_MAX,
    SHAPE_ALPHA_THR,        
    ASPECT_TOL,        
    PHASH_SHAPE_MAX,
    USE_SHAPE_CHECK,
    PHASH_COLOR_MAX,
    USE_COLOR_CHECK,
    PHASH_EDGE_MAX,
    USE_EDGE_CHECK,
    ROT_EARLYSTOP_SLACK,
    CONTENT_AREA_TOL,
    HGRAM_CHISQ_MAX
)

from ..stores.index_store import IndexStore
from ..stores.feature_store import FeatureStore
from ..stores.logger import ActionsLogger
from .group_widget import GroupResultsWidget

from .group_widget import GroupResultsWidget
from .widgets import ImageLabel, BBoxGraphicsView
from .dialogs import PairDecisionDialog as PairDialog
from ..utils.image_io import write_results

from ..utils.image_io import (
    qpixmap_from_rgba,
    read_image_rgba,
    make_thumb_rgba,
)

from ..core.phash import (
    phash_from_rgba,
    phash_from_canon_rgba,
    phash_from_canon_uv,
    phash_from_canon_alpha,
    phash_from_canon_edge,
    best_rot_hamming_fast,
    _hamming64,
)

from ..utils.image_io import trim_and_pad_rgba

from ..core.alpha_cc import (
    alpha_cc_boxes,
    is_spritesheet,
)

from ..core.features import (
    content_area_ratio,
    gray_hist32,
    crop_aspect_ratio,
    chisq_dist
)

from ..constants import (
    LIGHT_QSS, DARK_QSS,
    ROT_DEG_STEP_DEFAULT, INCLUDE_FLIP_TEST_DEFAULT,
    PHASH_HAMMING_MAX_DEFAULT, PHASH_HAMMING_MAX_INTRA_DEFAULT,
    ALPHA_THR_DEFAULT, MIN_AREA_DEFAULT, MIN_SIZE_DEFAULT,
    SPRITESHEET_MIN_SEGMENTS_DEFAULT, SPRITESHEET_MIN_COVERAGE_DEFAULT,
    SINGLES_BUCKET_SHIFT,
    SINGLES_GROUP_KEY,      
    CANON_PAD_PRIMARY,      
    CANON_PAD_SECONDARY,     
    SHAPE_ALPHA_THR,        
    ASPECT_TOL,        
)

try:
    from ..constants import ROT_EARLYSTOP_SLACK
except Exception:
    ROT_EARLYSTOP_SLACK = 0

from PIL import Image

class LRUCache:
    """簡單的 LRU 快取實作"""
    def __init__(self, capacity=50):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

@dataclass
class ImageItem:
    id: str
    src_path: Optional[str]     
    rgba: np.ndarray
    display_name: str         
    keep: Optional[bool] = None
    group_id: Optional[str] = None
    parent_uuid: Optional[str] = None         
    sub_id: Optional[str] = None             
    bbox: Optional[Tuple[int,int,int,int]] = None  

@dataclass
class PairHit:
    left_id: str
    right_id: str
    hamming: int

class ScanWorker(QtCore.QObject):
    progressInit = QtCore.pyqtSignal(int, int)
    progressStep = QtCore.pyqtSignal(int)
    finished     = QtCore.pyqtSignal(dict)
    logMessage   = QtCore.pyqtSignal(str)

    def __init__(self, task_args):
        super().__init__()
        self.task_args = task_args
        self._abort = False

    @QtCore.pyqtSlot()
    def run(self):
        """
        這是背景執行緒的進入點。
        """
        input_order = self.task_args.get("input_order", [])
        project_root = self.task_args.get("project_root")
        alpha_thr = self.task_args.get("alpha_thr")
        min_area = self.task_args.get("min_area")
        min_size = self.task_args.get("min_size")
        spr_min_segs = self.task_args.get("spr_min_segs")
        spr_min_cover = self.task_args.get("spr_min_cover")
        phash_hamming_max_intra = self.task_args.get("phash_hamming_max_intra")
        phash_hamming_max = self.task_args.get("phash_hamming_max")

        local_items_raw = []
        local_id2item = {}
        local_pool = []
        local_pairs = []
        local_seen_pair_keys = set()
        local_in_pair_ids = set()
        local_sheet_meta = {}
        local_json_payloads = {}
        
        features_store = FeatureStore(project_root)
        index_store = IndexStore(project_root)
        index_store.load()

        try:
            self.logMessage.emit("步驟 1/4: 讀取圖片...")
            total = len(input_order)
            if total == 0:
                self.finished.emit({"error": "No images to process."})
                return
                
            steps_read = total
            for idx, (uid, p) in enumerate(input_order):
                if self._abort: return
                try:
                    rgba = read_image_rgba(p)
                    name = os.path.basename(p)
                    item = ImageItem(id=uid, src_path=p, rgba=rgba, display_name=name)
                    local_items_raw.append(item)
                    local_id2item[uid] = item
                except Exception as e:
                    self.logMessage.emit(f"[錯誤] 讀取 {p} 失敗: {e}")

            self.logMessage.emit("步驟 2/4: 偵測並切割 Spritesheet...")
            steps_alpha = len(local_items_raw)
            for idx, it in enumerate(local_items_raw, 1):
                if self._abort: return
                boxes_strict = alpha_cc_boxes(it.rgba, alpha_thr, min_area, min_size)
                boxes_loose  = alpha_cc_boxes(it.rgba, alpha_thr, max(100, min_area // 2), max(4, min_size // 2))
                is_sheet_strict = is_spritesheet(it.rgba, boxes_strict, spr_min_segs, spr_min_cover)
                is_sheet_loose  = is_spritesheet(it.rgba, boxes_loose,  spr_min_segs, spr_min_cover)
                use_boxes = boxes_strict if is_sheet_strict else (boxes_loose if is_sheet_loose else None)

                if use_boxes is not None:
                    parent_uuid = it.id
                    rel_path = os.path.relpath(it.src_path, project_root) if project_root and it.src_path else it.display_name
                    local_sheet_meta[parent_uuid] = {
                        "source_path": rel_path,
                        "dimensions": {"width": int(it.rgba.shape[1]), "height": int(it.rgba.shape[0])},
                        "sub_images": []
                    }
                    for i, (x, y, w, h) in enumerate(use_boxes):
                        crop = trim_and_pad_rgba(it.rgba[y:y+h, x:x+w, :], pad=0)
                        sub_id_str = f"sub_{i}"
                        full_id = f"{parent_uuid}#{sub_id_str}"
                        sub = ImageItem(id=full_id, src_path=None, rgba=crop, display_name=f"{it.display_name}#{sub_id_str}",
                                        group_id=it.display_name, keep=None, parent_uuid=parent_uuid, sub_id=i, bbox=(x, y, w, h))
                        local_pool.append(sub); local_id2item[sub.id] = sub
                        local_sheet_meta[parent_uuid]["sub_images"].append(
                            {"sub_id": i, "bbox": [int(x), int(y), int(w), int(h)], "sub_uuid": sub.id}
                        )
                else:
                    it.keep = None
                    local_pool.append(it); local_id2item[it.id] = it

            self.logMessage.emit("步驟 3/4: 提取影像特徵...")
            N = len(local_pool)
            steps_feat = N
            steps_pairs = (N * (N - 1)) // 2
            total_steps = steps_read + steps_alpha + steps_feat + steps_pairs
            self.progressInit.emit(0, total_steps if total_steps > 0 else 1)
            
            done = steps_read + steps_alpha
            self.progressStep.emit(done)

            phash_primary, phash_secondary, phash_u, phash_v, phash_alpha, phash_edge = {}, {}, {}, {}, {}, {}
            area_map, hgram_map = {}, {}

            # --- 【修改點 1: 替換旗標】 ---
            # (不再使用 all_features_were_cached)
            # all_features_were_cached = True
            
            # (改用一個 set 來儲存乾淨的 item ID)
            clean_item_ids = set()

            for it in local_pool:
                if self._abort: return
                
                used_cache = False
                is_clean = False
                try:
                    # (it.id 是 uuid#sub_id, it.parent_uuid 才是母圖 uuid)
                    check_uuid = it.parent_uuid if it.parent_uuid else it.id
                    rel_path = index_store._uuid_to_rel.get(check_uuid)
                    if rel_path:
                        meta = index_store.data.get("image_map", {}).get(rel_path)
                        if meta and meta.get("uuid") == check_uuid:
                            is_clean = not meta.get("dirty_features", True)
                except Exception:
                    is_clean = False

                if is_clean:
                    cf = self._load_cached_features_for_item(it, features_store)
                    if cf and "phash_primary" in cf:
                        phash_primary[it.id]   = cf.get("phash_primary") or 0
                        phash_secondary[it.id] = cf.get("phash_secondary") or 0
                        phash_u[it.id]         = cf.get("phash_u") or 0
                        phash_v[it.id]         = cf.get("phash_v")
                        phash_alpha[it.id]     = cf.get("phash_alpha") or 0
                        phash_edge[it.id]      = cf.get("phash_edge") or 0
                        area_map[it.id]        = cf.get("area_ratio") or 0.0
                        hgram_list = cf.get("hgram_gray32")
                        if hgram_list is not None:
                            hgram_map[it.id] = np.array(hgram_list, dtype=np.float32)
                        else:
                            hgram_map[it.id] = np.zeros(32, dtype=np.float32)
                        used_cache = True
                        clean_item_ids.add(it.id) # <-- 儲存乾淨的 ID

                if not used_cache:
                    # (all_features_were_cached = False) <-- 移除
                    phash_primary[it.id]   = phash_from_canon_rgba(it.rgba, alpha_thr, pad_ratio=CANON_PAD_PRIMARY)
                    phash_secondary[it.id] = phash_from_canon_rgba(it.rgba, alpha_thr, pad_ratio=CANON_PAD_SECONDARY)
                    u, v = phash_from_canon_uv(it.rgba, alpha_thr, pad_ratio=CANON_PAD_PRIMARY)
                    phash_u[it.id], phash_v[it.id] = u, v
                    phash_alpha[it.id] = phash_from_canon_alpha(it.rgba, alpha_thr=SHAPE_ALPHA_THR, pad_ratio=CANON_PAD_SECONDARY)
                    phash_edge[it.id]  = phash_from_canon_edge(it.rgba, alpha_thr=SHAPE_ALPHA_THR, pad_ratio=CANON_PAD_PRIMARY)
                    area_map[it.id]  = content_area_ratio(it.rgba, alpha_thr)
                    hgram_map[it.id] = gray_hist32(it.rgba, alpha_thr)
                
                done += 1
                self.progressStep.emit(done)
            
            # --- 3b. 儲存特徵快取 ---
            # (此區塊不變 ... )
            temp_id_map = {i.id: i for i in local_pool}
            for item in temp_id_map.values():
                if item.parent_uuid is not None: continue 
                uuid_ = item.id
                hgram = hgram_map.get(uuid_)
                feat = {
                    "phash_primary": phash_primary.get(uuid_),"phash_secondary": phash_secondary.get(uuid_),
                    "phash_u": phash_u.get(uuid_),"phash_v": phash_v.get(uuid_),"phash_alpha": phash_alpha.get(uuid_),
                    "phash_edge": phash_edge.get(uuid_),"area_ratio": area_map.get(uuid_),
                    "hgram_gray32": hgram.tolist() if hgram is not None else None,
                }
                rel_path = None
                try:
                    rel_path = index_store._uuid_to_rel.get(uuid_)
                except Exception: pass
                if not rel_path and item.src_path:
                    try: rel_path = os.path.relpath(item.src_path, project_root)
                    except ValueError: rel_path = item.src_path
                payload = {
                    "uuid": uuid_,"source_path": rel_path or item.display_name,"is_spritesheet": False,
                    "dimensions": {"width": item.rgba.shape[1], "height": item.rgba.shape[0]},"features": feat,
                }
                local_json_payloads[uuid_] = payload
                features_store.save(uuid_, payload)
                index_store.mark_clean_by_uuid(uuid_) 
            for parent_uuid, meta in local_sheet_meta.items():
                updated_sub_images = []
                for sub_info in meta.get("sub_images", []):
                    sub_item_id = sub_info.get("sub_uuid")
                    if not sub_item_id: continue
                    hgram = hgram_map.get(sub_item_id)
                    sub_feat = {
                        "phash_primary": phash_primary.get(sub_item_id),"phash_secondary": phash_secondary.get(sub_item_id),
                        "phash_u": phash_u.get(sub_item_id),"phash_v": phash_v.get(sub_item_id),"phash_alpha": phash_alpha.get(sub_item_id),
                        "phash_edge": phash_edge.get(sub_item_id),"area_ratio": area_map.get(sub_item_id),
                        "hgram_gray32": hgram.tolist() if hgram is not None else None,
                    }
                    new_sub_info = {"sub_id": sub_info["sub_id"],"bbox": sub_info["bbox"],"features": sub_feat }
                    updated_sub_images.append(new_sub_info)
                mother_payload = {
                    "uuid": parent_uuid,"source_path": meta.get("source_path"),"is_spritesheet": True,
                    "dimensions": meta.get("dimensions"),"sub_images": updated_sub_images,"features": {} 
                }
                local_json_payloads[parent_uuid] = mother_payload
                features_store.save(parent_uuid, mother_payload)
                index_store.mark_clean_by_uuid(parent_uuid)
            index_store.save()
            # --- 儲存邏輯結束 ---

            # --- 【修改點 3: 替換 N² 快取檢查】 ---
            
            # ( 舊的 `if all_features_were_cached:` 檢查已被移除 )
            
            # 1. 載入舊的 results.json，並存為一個 O(1) 查找的 map
            #    我們假設 score 儲存的是 hamming 距離
            old_pairs_map = {}
            if len(clean_item_ids) > 0: # 只有在有乾淨 item 時才需要載入
                self.logMessage.emit("載入舊的比對結果...")
                try:
                    res_path = os.path.join(project_root, ".image_cache", "results.json")
                    if os.path.exists(res_path):
                        with open(res_path, "r", encoding="utf-8") as f:
                            old_results = json.load(f)
                        for p in old_results.get("pairs", []):
                            la, lb = p.get("left_id"), p.get("right_id")
                            if la and lb:
                                key = (la, lb) if la < lb else (lb, la)
                                # 假設 "score" 儲存的是 hamming 距離
                                old_pairs_map[key] = int(p.get("score", 100)) 
                except Exception:
                    pass # old_pairs_map 保持為空

            # --- 4. N² 配對 (現在是增量模式) ---
            self.logMessage.emit("步驟 4/4: 執行相似度比對 (增量模式)...")
            
            for i in range(N):
                if self._abort: return
                    
                for j in range(i + 1, N):
                    
                    done += 1
                    if done % 1000 == 0:
                        self.progressStep.emit(done)
                    
                    A, B = local_pool[i], local_pool[j]
                    aid, bid = A.id, B.id
                    key = (aid, bid) if aid < bid else (bid, aid)

                    # --- 【關鍵修改】 ---
                    # 檢查 A 和 B 是否 *都* 是乾淨的
                    is_A_clean = aid in clean_item_ids
                    is_B_clean = bid in clean_item_ids

                    if is_A_clean and is_B_clean:
                        # 如果兩者都乾淨，我們就從舊結果中查找
                        cached_hamming = old_pairs_map.get(key)
                        if cached_hamming is not None:
                            # 找到了！從快取加入
                            # (我們需要檢查 hamming 門檻，因為參數可能已改變)
                            same_group = (A.parent_uuid is not None and A.parent_uuid == B.parent_uuid)
                            th = phash_hamming_max_intra if same_group else phash_hamming_max
                            
                            if cached_hamming <= th:
                                local_pairs.append(PairHit(aid, bid, cached_hamming))
                                local_seen_pair_keys.add(key)
                                local_in_pair_ids.update([aid, bid])
                        
                        # (無論是否找到，都跳過昂貴的計算)
                        continue 
                    # --- 【修改結束】 ---

                    # (如果 A 或 B 至少有一個是 "dirty"，我們必須重新計算)
                    
                    same_group = (A.parent_uuid is not None and A.parent_uuid == B.parent_uuid)
                    th = phash_hamming_max_intra if same_group else phash_hamming_max
                    arA = crop_aspect_ratio(A.rgba, alpha_thr); arB = crop_aspect_ratio(B.rgba, alpha_thr)
                    if abs(np.log((arA + 1e-6) / (arB + 1e-6))) > ASPECT_TOL and not same_group:
                        continue

                    if abs(area_map[aid] - area_map[bid]) > CONTENT_AREA_TOL:
                        continue
                    
                    best, ang = best_rot_hamming_fast(phash_primary[aid], B.rgba, alpha_thr=alpha_thr, early_stop_at=th + ROT_EARLYSTOP_SLACK)
                    if best > th + ROT_EARLYSTOP_SLACK: 
                        continue
                    
                    if key in local_seen_pair_keys:
                        continue
                    
                    local_seen_pair_keys.add(key)
                    local_in_pair_ids.update([aid, bid])
                    local_pairs.append(PairHit(aid, bid, best))

            self.progressStep.emit(done)

            # --- 5. 寫入 results.json ---
            self.logMessage.emit("正在寫入結果...")
            if project_root:
                write_results(project_root, local_pairs, local_id2item)
            
            self.progressStep.emit(total_steps)

            # 6. 打包結果回傳
            results = {
                "error": None,
                "pairs": local_pairs,
                "id2item": local_id2item,
                "pool": local_pool,
                "items_raw": local_items_raw,
                "in_pair_ids": local_in_pair_ids,
                "seen_pair_keys": local_seen_pair_keys,

                "json_payloads": local_json_payloads
            }
            self.finished.emit(results)

        except Exception as e:
            import traceback
            err_msg = f"背景處理失敗: {e}\n{traceback.format_exc()}"
            self.logMessage.emit(f"[嚴重錯誤] {err_msg}")
            self.finished.emit({"error": err_msg})

    def abort(self):
        self._abort = True

    def _load_cached_features_for_item(self, it, features_store: FeatureStore) -> dict | None:
        """
        單張：直接讀 {uuid}.json 的 features
        子圖：讀母圖 {parent_uuid}.json -> sub_images[].features
        """
        try:
            if getattr(it, "parent_uuid", None) is None:
                # 這是散圖
                feat = features_store.load(it.id)
                return (feat or {}).get("features")
            else:
                # 這是子圖
                mother = features_store.load(it.parent_uuid)
                if not mother:
                    return None
                for si in (mother.get("sub_images") or []):
                    # 比較 int == int
                    if si.get("sub_id") == it.sub_id:
                        return (si.get("features") or {})
        except Exception as e:
            self.logMessage.emit(f"[Cache Error] Failed to load {it.id}: {e}")
            return None
        return None

# class ScanWorker(QtCore.QObject):
#     progressInit = QtCore.pyqtSignal(int, int)
#     progressStep = QtCore.pyqtSignal(int)
#     # 1. 修改 finished 信號，使其能回傳結果 (一個 dict)
#     finished     = QtCore.pyqtSignal(dict)
#     # 2. 增加一個日誌信號，回報目前狀態
#     logMessage   = QtCore.pyqtSignal(str)

#     def __init__(self, task_args):
#         super().__init__()
#         self.task_args = task_args
#         self._abort = False

#     @QtCore.pyqtSlot()
#     def run(self):
#         """
#         這是背景執行緒的進入點。
#         MainWindow.on_run() 的所有耗時邏輯都在這裡。
#         """
        
#         # 3. 從 task_args 解開所有需要的參數
#         input_order = self.task_args.get("input_order", [])
#         project_root = self.task_args.get("project_root")
#         alpha_thr = self.task_args.get("alpha_thr")
#         min_area = self.task_args.get("min_area")
#         min_size = self.task_args.get("min_size")
#         spr_min_segs = self.task_args.get("spr_min_segs")
#         spr_min_cover = self.task_args.get("spr_min_cover")
#         phash_hamming_max_intra = self.task_args.get("phash_hamming_max_intra")
#         phash_hamming_max = self.task_args.get("phash_hamming_max")

#         # 4. 準備本地變數 (儲存結果用)
#         local_items_raw = []
#         local_id2item = {}
#         local_pool = []
#         local_pairs = []
#         local_seen_pair_keys = set()
#         local_in_pair_ids = set()
#         local_sheet_meta = {}
#         local_json_payloads = {}
        
#         # (用於特徵快取)
#         features_store = FeatureStore(project_root)
#         index_store = IndexStore(project_root)
#         index_store.load()

#         try:
#             # === (以下是從 MainWindow.on_run 剪下並貼上的程式碼) ===
            
#             # --- 1. 讀取圖片 ---
#             self.logMessage.emit("步驟 1/4: 讀取圖片...")
#             total = len(input_order)
#             if total == 0:
#                 self.finished.emit({"error": "No images to process."})
#                 return
                
#             steps_read = total
#             # ( ... 這裡貼上 on_run 855-867 行的讀檔迴圈 ... )
#             # ( ... 並將 self.items_raw -> local_items_raw ... )
#             # ( ... self.id2item -> local_id2item ... )
#             for idx, (uid, p) in enumerate(input_order):
#                 if self._abort: return
#                 try:
#                     rgba = read_image_rgba(p)
#                     name = os.path.basename(p)
#                     item = ImageItem(id=uid, src_path=p, rgba=rgba, display_name=name)
#                     local_items_raw.append(item)
#                     local_id2item[uid] = item
#                 except Exception as e:
#                     self.logMessage.emit(f"[錯誤] 讀取 {p} 失敗: {e}")

#             # --- 2. 切割 Spritesheet ---
#             self.logMessage.emit("步驟 2/4: 偵測並切割 Spritesheet...")
#             steps_alpha = len(local_items_raw)
#             # ( ... 這裡貼上 on_run 873-909 行的 alpha_cc 邏輯 ... )
#             # ( ... 將 self.pool -> local_pool ... )
#             # ( ... self.id2item -> local_id2item ... )
#             # ( ... self.sheet_meta -> local_sheet_meta ... )
#             for idx, it in enumerate(local_items_raw, 1):
#                 if self._abort: return
#                 boxes_strict = alpha_cc_boxes(it.rgba, alpha_thr, min_area, min_size)
#                 boxes_loose  = alpha_cc_boxes(it.rgba, alpha_thr, max(100, min_area // 2), max(4, min_size // 2))
#                 is_sheet_strict = is_spritesheet(it.rgba, boxes_strict, spr_min_segs, spr_min_cover)
#                 is_sheet_loose  = is_spritesheet(it.rgba, boxes_loose,  spr_min_segs, spr_min_cover)
#                 use_boxes = boxes_strict if is_sheet_strict else (boxes_loose if is_sheet_loose else None)

#                 if use_boxes is not None:
#                     parent_uuid = it.id
#                     rel_path = os.path.relpath(it.src_path, project_root) if project_root and it.src_path else it.display_name
#                     local_sheet_meta[parent_uuid] = {
#                         "source_path": rel_path,
#                         "dimensions": {"width": int(it.rgba.shape[1]), "height": int(it.rgba.shape[0])},
#                         "sub_images": []
#                     }
#                     for i, (x, y, w, h) in enumerate(use_boxes):
#                         crop = trim_and_pad_rgba(it.rgba[y:y+h, x:x+w, :], pad=0)
#                         sub_id_str = f"sub_{i}"
#                         full_id = f"{parent_uuid}#{sub_id_str}"
#                         sub = ImageItem(id=full_id, src_path=None, rgba=crop, display_name=f"{it.display_name}#{sub_id_str}",
#                                         group_id=it.display_name, keep=None, parent_uuid=parent_uuid, sub_id=i, bbox=(x, y, w, h))
#                         local_pool.append(sub); local_id2item[sub.id] = sub
#                         local_sheet_meta[parent_uuid]["sub_images"].append(
#                             {"sub_id": i, "bbox": [int(x), int(y), int(w), int(h)], "sub_uuid": sub.id}
#                         )
#                 else:
#                     it.keep = None
#                     local_pool.append(it); local_id2item[it.id] = it

#             # --- 3. 特徵提取 ---
#             self.logMessage.emit("步驟 3/4: 提取影像特徵...")
#             N = len(local_pool)
#             steps_feat = N
#             steps_pairs = (N * (N - 1)) // 2
#             total_steps = steps_read + steps_alpha + steps_feat + steps_pairs
#             self.progressInit.emit(0, total_steps if total_steps > 0 else 1)
            
#             done = steps_read + steps_alpha
#             self.progressStep.emit(done)

#             phash_primary, phash_secondary, phash_u, phash_v, phash_alpha, phash_edge = {}, {}, {}, {}, {}, {}
#             area_map, hgram_map = {}, {}

#             # ( ... 這裡貼上 on_run 931-984 行的特徵提取邏輯 ... )
#             # ( ... 為了簡化，快取邏輯 (if used_cache) 先移除 ... )
#             # ( ... 全部重新計算 ... )
#             for it in local_pool:
#                 if self._abort: return
                
#                 used_cache = False
                
#                 # 1. 檢查 IndexStore 看檔案是否"乾淨"
#                 is_clean = False
#                 try:
#                     rel_path = index_store._uuid_to_rel.get(it.id) or index_store._uuid_to_rel.get(it.parent_uuid)
#                     if rel_path:
#                         meta = index_store.data.get("image_map", {}).get(rel_path)
#                         if meta and meta.get("uuid") in (it.id, it.parent_uuid):
#                             is_clean = not meta.get("dirty_features", True)
#                 except Exception:
#                     is_clean = False

#                 # 2. 如果乾淨，嘗試讀取
#                 if is_clean:
#                     cf = self._load_cached_features_for_item(it, features_store) # 傳入 features_store
#                     if cf and "phash_primary" in cf:
#                         # 1. 為所有 pHash 提供預設值 0
#                         phash_primary[it.id]   = cf.get("phash_primary") or 0
#                         phash_secondary[it.id] = cf.get("phash_secondary") or 0
#                         phash_u[it.id]         = cf.get("phash_u") or 0
#                         phash_v[it.id]         = cf.get("phash_v")
#                         phash_alpha[it.id]     = cf.get("phash_alpha") or 0
#                         phash_edge[it.id]      = cf.get("phash_edge") or 0
                        
#                         # 2. 為 area_ratio 提供預設值 0.0 (修復崩潰)
#                         area_map[it.id]        = cf.get("area_ratio") or 0.0
                        
#                         # 3. 為 hgram 提供一個空的 np 陣列，而不是 None
#                         hgram_list = cf.get("hgram_gray32")
#                         if hgram_list is not None:
#                             hgram_map[it.id] = np.array(hgram_list, dtype=np.float32)
#                         else:
#                             hgram_map[it.id] = np.zeros(32, dtype=np.float32) # 使用 0 陣列
                        
#                         used_cache = True

#                 # 3. 如果沒有快取或快取"髒了"，才重新計算
#                 if not used_cache:
#                     phash_primary[it.id]   = phash_from_canon_rgba(it.rgba, alpha_thr, pad_ratio=CANON_PAD_PRIMARY)
#                     phash_secondary[it.id] = phash_from_canon_rgba(it.rgba, alpha_thr, pad_ratio=CANON_PAD_SECONDARY)
#                     u, v = phash_from_canon_uv(it.rgba, alpha_thr, pad_ratio=CANON_PAD_PRIMARY)
#                     phash_u[it.id], phash_v[it.id] = u, v
#                     phash_alpha[it.id] = phash_from_canon_alpha(it.rgba, alpha_thr=SHAPE_ALPHA_THR, pad_ratio=CANON_PAD_SECONDARY)
#                     phash_edge[it.id]  = phash_from_canon_edge(it.rgba, alpha_thr=SHAPE_ALPHA_THR, pad_ratio=CANON_PAD_PRIMARY)
#                     area_map[it.id]  = content_area_ratio(it.rgba, alpha_thr)
#                     hgram_map[it.id] = gray_hist32(it.rgba, alpha_thr)
                
#                 done += 1
#                 self.progressStep.emit(done)
            
#             # --- 3b. 儲存特徵快取 (!!! 這是貼上的程式碼 !!!) ---
#             temp_id_map = {i.id: i for i in local_pool}

#             # 儲存 散圖 (Single Images)
#             for item in temp_id_map.values():
#                 if item.parent_uuid is not None:
#                     continue 
                
#                 uuid_ = item.id
#                 # (我們假設所有 features 都是新算的，所以總是儲存)
#                 # (註解掉 mark_clean_by_uuid 檢查，強制寫入)
#                 # if not index_store.mark_clean_by_uuid(uuid_):
#                 #     continue 

#                 hgram = hgram_map.get(uuid_)
#                 feat = {
#                     "phash_primary": phash_primary.get(uuid_),
#                     "phash_secondary": phash_secondary.get(uuid_),
#                     "phash_u": phash_u.get(uuid_),
#                     "phash_v": phash_v.get(uuid_),
#                     "phash_alpha": phash_alpha.get(uuid_),
#                     "phash_edge": phash_edge.get(uuid_),
#                     "area_ratio": area_map.get(uuid_),
#                     "hgram_gray32": hgram.tolist() if hgram is not None else None,
#                 }
                
#                 # 試圖從 index_store 取得 rel_path
#                 rel_path = None
#                 try:
#                     rel_path = index_store._uuid_to_rel.get(uuid_)
#                     if rel_path:
#                         meta = index_store.data.get("image_map", {}).get(rel_path) or {}
#                 except Exception:
#                     meta = {}
                
#                 if not rel_path and item.src_path:
#                     try:
#                         rel_path = os.path.relpath(item.src_path, project_root)
#                     except ValueError:
#                         rel_path = item.src_path

#                 payload = {
#                     "uuid": uuid_,
#                     "source_path": rel_path or item.display_name,
#                     "is_spritesheet": False,
#                     "dimensions": {"width": item.rgba.shape[1], "height": item.rgba.shape[0]},
#                     "features": feat,
#                 }
#                 local_json_payloads[uuid_] = payload
#                 features_store.save(uuid_, payload)
#                 index_store.mark_clean_by_uuid(uuid_) # 儲存後標記為乾淨

#             # 儲存 組圖 (Spritesheets)
#             for parent_uuid, meta in local_sheet_meta.items():
#                 # (強制儲存)
#                 # if not index_store.mark_clean_by_uuid(parent_uuid):
#                 #     continue
                
#                 updated_sub_images = []
#                 for sub_info in meta.get("sub_images", []):
#                     sub_item_id = sub_info.get("sub_uuid")
#                     if not sub_item_id:
#                         continue
                    
#                     hgram = hgram_map.get(sub_item_id)
#                     sub_feat = {
#                         "phash_primary": phash_primary.get(sub_item_id),
#                         "phash_secondary": phash_secondary.get(sub_item_id),
#                         "phash_u": phash_u.get(sub_item_id),
#                         "phash_v": phash_v.get(sub_item_id),
#                         "phash_alpha": phash_alpha.get(sub_item_id),
#                         "phash_edge": phash_edge.get(sub_item_id),
#                         "area_ratio": area_map.get(sub_item_id),
#                         "hgram_gray32": hgram.tolist() if hgram is not None else None,
#                     }
                    
#                     new_sub_info = {
#                         "sub_id": sub_info["sub_id"],
#                         "bbox": sub_info["bbox"],
#                         "features": sub_feat 
#                     }
#                     updated_sub_images.append(new_sub_info)

#                 mother_payload = {
#                     "uuid": parent_uuid,
#                     "source_path": meta.get("source_path"),
#                     "is_spritesheet": True,
#                     "dimensions": meta.get("dimensions"),
#                     "sub_images": updated_sub_images,
#                     "features": {} # 母圖本身不儲存特徵
#                 }
#                 local_json_payloads[parent_uuid] = mother_payload
#                 features_store.save(parent_uuid, mother_payload)
#                 index_store.mark_clean_by_uuid(parent_uuid) # 儲存後標記為乾淨
            
#             # 確保 index.json 被寫回
#             index_store.save()
#             # --- 儲存邏輯結束 ---

#             # --- 4. N² 配對 ---
#             self.logMessage.emit("步驟 4/4: 執行相似度比對...")
            
#             for i in range(N):
#                 if self._abort: return
                
#                 # (移除 i 迴圈的進度更新)
#                 # if i % 100 == 0: ...
                    
#                 for j in range(i + 1, N):
                    
#                     # --- 修正開始 ---
#                     # 1. 將 done += 1 移到 J 迴圈的最頂部
#                     #    確保「每一次比對」都會推進度條
#                     done += 1
                    
#                     # 2. (可選) 為了效能，不要 300 萬次都更新 UI
#                     #    改成每 1000 次比對才更新一次進度條
#                     if done % 1000 == 0:
#                         self.progressStep.emit(done)
#                     # --- 修正結束 ---
                    
#                     A, B = local_pool[i], local_pool[j]
#                     aid, bid = A.id, B.id
                    
#                     # ( ... 接下來是 TIER 1, 2, 3 的所有過濾邏輯 ... )
#                     same_group = (A.parent_uuid is not None and A.parent_uuid == B.parent_uuid)
#                     th = phash_hamming_max_intra if same_group else phash_hamming_max
#                     arA = crop_aspect_ratio(A.rgba, alpha_thr); arB = crop_aspect_ratio(B.rgba, alpha_thr)
#                     if abs(np.log((arA + 1e-6) / (arB + 1e-6))) > ASPECT_TOL and not same_group:
#                         continue # (現在 continue 之前已經計過數了)

#                     if abs(area_map[aid] - area_map[bid]) > CONTENT_AREA_TOL:
#                         continue # (現在 continue 之前已經計過數了)
                    
#                     # ( ... TIER 3: best_rot_hamming_fast ... )
#                     best, ang = best_rot_hamming_fast(phash_primary[aid], B.rgba, alpha_thr=alpha_thr, early_stop_at=th + ROT_EARLYSTOP_SLACK)
#                     if best > th + ROT_EARLYSTOP_SLACK: 
#                         continue # (現在 continue 之前已經計過數了)
                    
#                     key = (aid, bid) if aid < bid else (bid, aid)
#                     if key in local_seen_pair_keys:
#                         continue # (現在 continue 之前已經計過數了)
                    
#                     local_seen_pair_keys.add(key)
#                     local_in_pair_ids.update([aid, bid])
#                     local_pairs.append(PairHit(aid, bid, best))
                    
#                     # (移除 j 迴圈底部的 done += 1)

#             # (迴圈結束後，發送一次最終進度)
#             self.progressStep.emit(done)

#             # --- 5. 寫入 results.json ---
#             self.logMessage.emit("正在寫入結果...")
#             if project_root:
#                 write_results(project_root, local_pairs, local_id2item)
            
#             self.progressStep.emit(total_steps)

#             # 6. 打包結果回傳
#             results = {
#                 "error": None,
#                 "pairs": local_pairs,
#                 "id2item": local_id2item,
#                 "pool": local_pool,
#                 "items_raw": local_items_raw,
#                 "in_pair_ids": local_in_pair_ids,
#                 "seen_pair_keys": local_seen_pair_keys,

#                 "json_payloads": local_json_payloads
#             }
#             self.finished.emit(results)

#         except Exception as e:
#             import traceback
#             err_msg = f"背景處理失敗: {e}\n{traceback.format_exc()}"
#             self.logMessage.emit(f"[嚴重錯誤] {err_msg}")
#             self.finished.emit({"error": err_msg})

#     def abort(self):
#         self._abort = True

#     def _load_cached_features_for_item(self, it, features_store: FeatureStore) -> dict | None:
#         """
#         單張：直接讀 {uuid}.json 的 features
#         子圖：讀母圖 {parent_uuid}.json -> sub_images[].features
#         """
#         try:
#             if getattr(it, "parent_uuid", None) is None:
#                 # 這是散圖
#                 feat = features_store.load(it.id)
#                 return (feat or {}).get("features")
#             else:
#                 # 這是子圖
#                 mother = features_store.load(it.parent_uuid)
#                 if not mother:
#                     return None
#                 for si in (mother.get("sub_images") or []):
#                     # 比較 int == int
#                     if si.get("sub_id") == it.sub_id:
#                         return (si.get("features") or {})
#         except Exception as e:
#             self.logMessage.emit(f"[Cache Error] Failed to load {it.id}: {e}")
#             return None
#         return None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_root = None
        self.index = None
        self.features = None
        self.logger = None
        self.worker_thread = None
        self.worker = None

        self.thumb_json_cache = {}
        self.thumb_pixmap_cache = {}
        self.thumb_scaled_base_cache = LRUCache(capacity=50)
        self.all_json_payloads = {}

        self.setWindowTitle(f"Sprite Dedupe {VERSION}")
        self.resize(1260, 780)

        self.rot_step = ROT_DEG_STEP_DEFAULT
        self.include_flip = INCLUDE_FLIP_TEST_DEFAULT
        self.phash_hamming_max_intra = PHASH_HAMMING_MAX_INTRA_DEFAULT

        self.items_raw: List['ImageItem'] = []
        self.group_nodes = {}
        self.pool: List['ImageItem'] = []
        self.id2item: Dict[str, 'ImageItem'] = {}
        self.pairs: List['PairHit'] = []
        self.seen_pair_keys = set()
        self.in_pair_ids = set()

        self.phash_hamming_max = PHASH_HAMMING_MAX_DEFAULT
        self.alpha_thr = ALPHA_THR_DEFAULT
        self.min_area = MIN_AREA_DEFAULT
        self.min_size = MIN_SIZE_DEFAULT
        self.spr_min_segs = SPRITESHEET_MIN_SEGMENTS_DEFAULT
        self.spr_min_cover = SPRITESHEET_MIN_COVERAGE_DEFAULT

        self.dark_mode = False
        self.temp_dir = tempfile.mkdtemp(prefix="sprite_pro_")

        self.group_view = GroupResultsWidget(self)
        self.group_view.request_pair_decision.connect(self._open_pair_dialog_for_uuids)

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._apply_theme()

        self._input_order = []
        self._input_paths = set()
        self._list_mode = "input"

        self.member_to_groups = {}
        self.obj_groups = {}

        self.list_files.itemSelectionChanged.connect(self._on_file_selected)

        # 進度條初始化
        self.progress.setRange(0, 0)
        self.progress.setValue(0)


    def _update_info_panel(self, uuid_: str | None, sub_id: int | None, group_id: str | None):
        """將目前選擇的成員或群組寫到右側 infoPanel。"""
        if not self._info_labels:
            return
        lab_uuid, lab_child, lab_size, lab_origin, lab_path, lab_phash, lab_dups = self._info_labels
        def set_(w, v): w.setText(str(v) if w else "-")

        if uuid_ is None and group_id:
            grp = next((g for g in (self.groups or []) if g.get("group_id") == group_id), None)
            mems = grp.get("members", []) if grp else []
            set_(lab_uuid,  "-")
            set_(lab_child, "-")
            set_(lab_size,  "-")
            set_(lab_origin, "群組")
            set_(lab_path,   group_id)
            set_(lab_phash,  "-")
            set_(lab_dups,   len(mems) if mems else "-")
            return

        if not uuid_:
            for w in (lab_uuid, lab_child, lab_size, lab_origin, lab_path, lab_phash, lab_dups):
                set_(w, "-")
            return

        feat = self._load_feat(uuid_) or {}
        set_(lab_uuid, uuid_)
        set_(lab_child, sub_id if sub_id is not None else "-")

        if sub_id is None:
            dims = feat.get("dimensions") or {}
            set_(lab_size, f'{dims.get("width","-")}×{dims.get("height","-")}')
            rel = feat.get("source_path")
            set_(lab_origin, "散圖")
            if rel:
                full_path = rel if os.path.isabs(rel) else os.path.join(self.project_root, rel)
                set_(lab_path, full_path)
            else:
                set_(lab_path, "-")
        else:
            set_(lab_origin, "組圖")
            pu = uuid_ 
            if pu:
                mother = self._load_feat(pu) or {}
                w = h = "-"
                for si in (mother.get("sub_images") or []):
                    if si.get("sub_id") == sub_id:
                        x, y, w, h = si.get("bbox", (0, 0, 0, 0))
                        break
                set_(lab_size, f"{w}×{h}")
                rel = mother.get("source_path")
                if rel:
                    full_path = rel if os.path.isabs(rel) else os.path.join(self.project_root, rel)
                    set_(lab_path, full_path)
                else:
                    set_(lab_path, "-")
            else:
                set_(lab_size, "-")
                set_(lab_path, "-")

        if group_id:
            grp = next((g for g in (self.groups or []) if g.get("group_id") == group_id), None)
            set_(lab_dups, len(grp.get("members", [])) if grp else "-")
        else:
            set_(lab_dups, "-")

        set_(lab_phash, "-")

    def _on_image_label_clicked(self, meta: dict):
        """處理在橫向群組中圖片縮圖的點擊事件"""
        uuid_  = meta.get("uuid")
        sub_id = meta.get("sub_id") 
        if not uuid_:
            return

        if hasattr(self, "group_view") and self.group_view:
            # <-- 【修改】同時傳入 sub_id
            self.group_view.select_member_by_uuid(uuid_, sub_id)

    def _thumbnail_from_path(self, abs_path: str, max_wh: int = 128) -> QtGui.QPixmap:
        pm = QtGui.QPixmap(abs_path)
        if pm.isNull():
            pm = QtGui.QPixmap(max_wh, max_wh)
            pm.fill(QtCore.Qt.darkGray)
            return pm
        return pm.scaled(max_wh, max_wh, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def _refresh_file_list_input_mode(self):
        """
        以「加入順序」更新左側清單。
        - 若是 QTableWidget：用 1 欄 N 列，列內放一個小容器（縮圖＋檔名）
        - 若還是 QListWidget：退回舊作法，不崩潰
        """
        data = self._input_order or []
        count = len(data)
        self.lb_count.setText(f"已加入：{count}")

        if isinstance(self.list_files, QtWidgets.QTableWidget):
            self.list_files.clearContents()
            self.list_files.setRowCount(count)
            for row, (uuid_, rel) in enumerate(data):
                cont = QtWidgets.QWidget()
                cont.setObjectName("cellContainer")
                h = QtWidgets.QHBoxLayout(cont)
                h.setContentsMargins(8, 6, 8, 6)
                h.setSpacing(8)

                abs_path = rel if os.path.isabs(rel) else os.path.join(self.project_root or "", rel)
                pm = self._thumbnail_from_path(abs_path, max_wh=48)

                lb_thumb = QtWidgets.QLabel()
                lb_thumb.setPixmap(pm)                        
                lb_thumb.setFixedSize(48, 48)
                lb_thumb.setScaledContents(False)             
                lb_thumb.setAlignment(Qt.AlignCenter)         
                lb_thumb.setProperty("role", "thumb")
                h.addWidget(lb_thumb, 0)

                name = os.path.basename(rel)
                lb = QtWidgets.QLabel(name)
                lb.setToolTip(rel)
                h.addWidget(lb, 1)

                self.list_files.setCellWidget(row, 0, cont)

                it = QtWidgets.QTableWidgetItem()
                it.setData(Qt.UserRole, {"uuid": uuid_, "rel": rel})
                self.list_files.setItem(row, 0, it)

            self.list_files.resizeRowsToContents()
            return

        if isinstance(self.list_files, QtWidgets.QListWidget):
            self.list_files.clear()
            for (uuid_, rel) in data:
                it = QtWidgets.QListWidgetItem(os.path.basename(rel))
                it.setData(Qt.UserRole, {"uuid": uuid_, "rel": rel})
                self.list_files.addItem(it)

    def _on_file_selected(self):
        if self._list_mode != "input":
            return

        items = self.list_files.selectedItems()
        if not items:
            if hasattr(self, "group_view"):
                if hasattr(self.group_view, "leftView"):
                    self.group_view.leftView.clear()
                if hasattr(self.group_view, "rightView"):
                    self.group_view.rightView.clear()
            self._update_info_panel(None, None, None)
            return

        it   = items[0]
        meta = it.data(QtCore.Qt.UserRole) or {}
        uuid_ = meta.get("uuid")
        if not uuid_:
            return

        info = {}
        try:
            feat_p = os.path.join(self.project_root, ".image_cache", "features", f"{uuid_}.json")
            if os.path.exists(feat_p):
                with open(feat_p, "r", encoding="utf-8") as f:
                    info = json.load(f)
        except Exception:
            info = {}

        dims = (info.get("dimensions") or {})
        w, h = dims.get("width"), dims.get("height")

        parent_uuid = info.get("parent_uuid")
        sub_id      = info.get("sub_id")
        rel         = info.get("source_path")

        mother = {}
        if parent_uuid:
            try:
                mpath = os.path.join(self.project_root, ".image_cache", "features", f"{parent_uuid}.json")
                if os.path.exists(mpath):
                    with open(mpath, "r", encoding="utf-8") as f:
                        mother = json.load(f)
            except Exception:
                mother = {}

            if sub_id is None:
                for si in mother.get("sub_images") or []:
                    if si.get("uuid") == uuid_:
                        sub_id = si.get("sub_id")
                        break

            if not rel:
                rel = mother.get("source_path")

        if hasattr(self, "group_view") and self.group_view.project_root:
            self.group_view._show_member_views(uuid_, sub_id if parent_uuid else None)

            self.group_view._last_selected_meta = {
                "type": "member",
                "uuid": uuid_,
                "sub_id": sub_id if parent_uuid else None,
            }

            if hasattr(self.group_view, "select_member_by_uuid"):
                self.group_view.select_member_by_uuid(uuid_, sub_id)

            btn = getattr(self.group_view, "btn_open_folder", None)
            if btn:
                btn.setEnabled(bool(rel))

        img_type = "子圖" if parent_uuid else "散圖"
        self.group_view._update_info_panel(uuid_, sub_id if parent_uuid else None, None)

        # if hasattr(self, "group_view"):
        #     if hasattr(self.group_view, "info_uuid"):   self.group_view.info_uuid.setText(uuid_[:8] if uuid_ else "-")
        #     if hasattr(self.group_view, "info_subid"):  self.group_view.info_subid.setText(str(sub_id) if sub_id is not None else "-")
        #     if hasattr(self.group_view, "info_source"): self.group_view.info_source.setText(rel or "-")
        #     if hasattr(self.group_view, "info_size"):   self.group_view.info_size.setText(f'{w if w else "-"}×{h if h else "-"}')
        #     if hasattr(self.group_view, "info_path"):   self.group_view.info_path.setText(rel or "-")

    def features_iter(self):
        """
        逐一讀取 .image_cache/features/*.json
        產出 (uuid, feature_dict)
        """
        feats_dir = None
        if getattr(self, "features", None) and getattr(self.features, "dir", None):
            feats_dir = self.features.dir
        elif getattr(self, "project_root", None):
            feats_dir = os.path.join(self.project_root, ".image_cache", "features")

        if not feats_dir or not os.path.isdir(feats_dir):
            return
        for fn in sorted(os.listdir(feats_dir)):
            if not fn.endswith(".json"):
                continue
            uuid_ = os.path.splitext(fn)[0]
            fp = os.path.join(feats_dir, fn)

            data = self.thumb_json_cache.get(uuid_)

            if data is None:
                # 2. 若快取沒有，才讀取 I/O
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.thumb_json_cache[uuid_] = data # 3. 存入快取
                except Exception:
                    continue
            
            data.setdefault("uuid", uuid_)
            yield uuid_, data

    def _load_feature_json(self, uuid_: str) -> dict | None:
        if not self.project_root: return None
        p = os.path.join(self.project_root, ".image_cache", "features", f"{uuid_}.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    @staticmethod
    def _get_sig(obj: dict | None) -> str | None:
        """
        從 feature 或 sub_feature 物件取『語意簽章』：
        - 若 signature 是 dict：優先 semantic，其次 label/name
        - 若 signature 是 str：直接回傳
        - 否則回 None（交由 _visual_bucket 當後備）
        """
        if not isinstance(obj, dict):
            return None
        src = obj.get("features", obj)
        sig = src.get("signature")
        if isinstance(sig, dict):
            return sig.get("semantic") or sig.get("label") or sig.get("name")
        if isinstance(sig, str) and sig.strip():
            return sig.strip()
        return None

    @staticmethod
    def _visual_bucket(obj: dict | None, shift: int = SINGLES_BUCKET_SHIFT) -> tuple | None:
        """
        視覺後備：以各種 phash 通道右移若干位形成粗 key，避免把不相干圖湊在一起。
        回傳 tuple（多通道一起當 key）；若完全沒有 phash 欄位則回 None。
        """
        if not isinstance(obj, dict):
            return None
        
        src = obj.get("features", obj)
        keys_try = [
            "phash_primary",
            "phash_secondary",
            "phash_u",
            "phash_v",
            "phash_alpha",
            "phash_edge",
        ]
        
        vals = []
        for k in keys_try:
            v = src.get(k)
            try:
                if v is None:
                    continue
                iv = int(v)
                vals.append(iv >> shift)
            except Exception:
                continue
        return tuple(vals) if vals else None

    def _get_full_id(self, uuid_, sub_id):
        """從 uuid 和 sub_id 組合出完整的 id (e.g., uuid#sub_id)"""
        if sub_id is not None:
            return f"{uuid_}#sub_{sub_id}"
        return uuid_

    # app/ui/main_window.py (替換第 896 行開始的函式)

    def _build_object_groups(self):
        """
        依『物件鍵』分群。物件鍵優先用 signature；沒有 signature 時退回 phash 視覺桶。
        只建立在全專案中出現次數 >= 2 的群（避免單一小圖也被分群）。
        會填好：
        self.object_groups: dict[group_key] -> list[{"uuid","sub_id","bbox","is_sheet"}]
        self.member_to_groups: dict[(uuid, sub_id)] -> set(group_key)
        """

        MIN_GROUP_OCCURRENCE = 2
        BUCKET_DROP_BITS = 8

        def _get_sig(d):
            s = d.get("signature")    
            if isinstance(s, dict):
                sig = s.get("semantic") or s.get("label") or s.get("name")
            elif isinstance(s, str):
                sig = s
            else:
                sig = None
            if isinstance(sig, str):
                sig = sig.strip().lower()
            return sig

        def _visual_bucket(obj: dict | None, shift: int = SINGLES_BUCKET_SHIFT) -> tuple | None:
            """
            視覺後備：以各種 phash 通道右移若干位形成粗 key，避免把不相干圖湊在一起。
            回傳 tuple（多通道一起當 key）；若完全沒有 phash 欄位則回 None。
            """
            if not isinstance(obj, dict):
                return None
            
            src = obj.get("features", obj)

            keys_try_new = [
                "phash_primary",
                "phash_secondary",
                "phash_u",
                "phash_v",
                "phash_alpha",
                "phash_edge",
            ]
            vals_new = []
            for k in keys_try_new:
                v = src.get(k)
                try:
                    if v is None:
                        continue
                    iv = int(v)
                    vals_new.append(iv >> shift)
                except Exception:
                    continue
            
            if vals_new:
                return tuple(vals_new)

            keys_try_old = [
                "phash",
                "phash_rgba",
            ]
            vals_old = []
            for k in keys_try_old:
                v = src.get(k)
                try:
                    if v is None:
                        continue
                    iv = int(v)
                    vals_old.append(iv >> shift)
                except Exception:
                    continue

            return tuple(vals_old) if vals_old else None

        key_counts = defaultdict(int)
        for _, feat in self.features_iter():
            k = _get_sig(feat) or _visual_bucket(feat)
            if k: key_counts[k] += 1
            for sub in (feat.get("sub_images") or []):
                sk = _get_sig(sub) or _visual_bucket(sub)
                if sk: key_counts[sk] += 1

        self.object_groups = {}
        self.member_to_groups = {}

        def _add(gk, uuid_, sub_id, bbox, is_sheet):
            self.object_groups.setdefault(gk, []).append(
                {"uuid": uuid_, "sub_id": sub_id, "bbox": bbox, "is_sheet": is_sheet}
            )
            self.member_to_groups.setdefault((uuid_, sub_id), set()).add(gk)

        for uuid_, feat in self.features_iter():
            k = _get_sig(feat) or _visual_bucket(feat)
            if k and key_counts[k] >= MIN_GROUP_OCCURRENCE:
                _add(k, uuid_, None, None, False)
            for i, sub in enumerate(feat.get("sub_images") or []):
                sk = _get_sig(sub) or _visual_bucket(sub)
                if sk and key_counts[sk] >= MIN_GROUP_OCCURRENCE:
                    sid = sub.get("sub_id")
                    if sid is not None and isinstance(sid, str) and sid.startswith("sub_"):
                        try:
                            sid = int(sid.split("_")[1])
                        except(IndexError, ValueError):
                            sid = i
                    else:
                        sid = i
                    _add(sk, uuid_, sid, sub.get("bbox"), True)

        assigned = set()
        ordered_groups = list(self.object_groups.items())

        new_groups = {}
        for gk, lst in ordered_groups:
            kept = []
            for m in lst:
                key = (m["uuid"], m.get("sub_id"))
                if key in assigned:
                    continue
                assigned.add(key)
                kept.append(m)
            if kept:
                new_groups[gk] = kept

        self.object_groups = new_groups

    # def _build_object_groups(self):
    #     """
    #     依『物件鍵』分群。物件鍵優先用 signature；沒有 signature 時退回 phash 視覺桶。
    #     只建立在全專案中出現次數 >= 2 的群（避免單一小圖也被分群）。
    #     會填好：
    #     self.object_groups: dict[group_key] -> list[{"uuid","sub_id","bbox","is_sheet"}]
    #     self.member_to_groups: dict[(uuid, sub_id)] -> set(group_key)
    #     """

    #     MIN_GROUP_OCCURRENCE = 2
    #     BUCKET_DROP_BITS = 8

    #     def _get_sig(d):
    #         s = d.get("signature")    
    #         if isinstance(s, dict):
    #             sig = s.get("semantic") or s.get("label") or s.get("name")
    #         elif isinstance(s, str):
    #             sig = s
    #         else:
    #             sig = None
    #         if isinstance(sig, str):
    #             sig = sig.strip().lower()
    #         return sig

    #     def _visual_bucket(obj: dict | None, shift: int = SINGLES_BUCKET_SHIFT) -> tuple | None:
    #         """
    #         視覺後備：以各種 phash 通道右移若干位形成粗 key，避免把不相干圖湊在一起。
    #         回傳 tuple（多通道一起當 key）；若完全沒有 phash 欄位則回 None。
    #         """
    #         if not isinstance(obj, dict):
    #             return None
            
    #         src = obj.get("features", obj)

    #         keys_try_new = [
    #             "phash_primary",
    #             "phash_secondary",
    #             "phash_u",
    #             "phash_v",
    #             "phash_alpha",
    #             "phash_edge",
    #         ]
    #         vals_new = []
    #         for k in keys_try_new:
    #             v = src.get(k)
    #             try:
    #                 if v is None:
    #                     continue
    #                 iv = int(v)
    #                 vals_new.append(iv >> shift)
    #             except Exception:
    #                 continue
            
    #         if vals_new:
    #             return tuple(vals_new)

    #         keys_try_old = [
    #             "phash",
    #             "phash_rgba",
    #         ]
    #         vals_old = []
    #         for k in keys_try_old:
    #             v = src.get(k)
    #             try:
    #                 if v is None:
    #                     continue
    #                 iv = int(v)
    #                 vals_old.append(iv >> shift)
    #             except Exception:
    #                 continue

    #         return tuple(vals_old) if vals_old else None

    #     key_counts = defaultdict(int)
    #     for _, feat in self.features_iter():
    #         k = _get_sig(feat) or _visual_bucket(feat)
    #         if k: key_counts[k] += 1
    #         for sub in (feat.get("sub_images") or []):
    #             sk = _get_sig(sub) or _visual_bucket(sub)
    #             if sk: key_counts[sk] += 1

    #     self.object_groups = {}
    #     self.member_to_groups = {}

    #     def _add(gk, uuid_, sub_id, bbox, is_sheet):
    #         self.object_groups.setdefault(gk, []).append(
    #             {"uuid": uuid_, "sub_id": sub_id, "bbox": bbox, "is_sheet": is_sheet}
    #         )
    #         self.member_to_groups.setdefault((uuid_, sub_id), set()).add(gk)

    #     for uuid_, feat in self.features_iter():
    #         k = _get_sig(feat) or _visual_bucket(feat)
    #         if k and key_counts[k] >= MIN_GROUP_OCCURRENCE:
    #             _add(k, uuid_, None, None, False)
    #         for i, sub in enumerate(feat.get("sub_images") or []):
    #             sk = _get_sig(sub) or _visual_bucket(sub)
    #             if sk and key_counts[sk] >= MIN_GROUP_OCCURRENCE:
    #                 sid = sub.get("sub_id")
    #                 if sid is not None and isinstance(sid, str) and sid.startswith("sub_"):
    #                     try:
    #                         sid = int(sid.split("_")[1])
    #                     except(IndexError, ValueError):
    #                         sid = i
    #                 else:
    #                     sid = i
    #                 _add(sk, uuid_, sid, sub.get("bbox"), True)

    #     assigned = set()
    #     ordered_groups = list(self.object_groups.items())

    #     new_groups = {}
    #     for gk, lst in ordered_groups:
    #         kept = []
    #         for m in lst:
    #             key = (m["uuid"], m.get("sub_id"))
    #             if key in assigned:
    #                 continue
    #             assigned.add(key)
    #             kept.append(m)
    #         if kept:
    #             new_groups[gk] = kept

    #     self.object_groups = new_groups


    def _group_color(self, name: str) -> QtGui.QColor:
        h = (hash(name) % 360)
        c = QtGui.QColor.fromHsv(h, 160, 230)
        return c

    def _apply_white_key(self, pix: QtGui.QPixmap, thr: int = 250) -> QtGui.QPixmap:
        """
        [從 widgets.py 複製]
        將接近白色的像素轉為透明：r>=thr 且 g>=thr 且 b>=thr（且原本 a==255）。
        """
        img = pix.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return pix

        bpl = img.bytesPerLine()
        ptr = img.bits()
        ptr.setsize(img.byteCount())      
        buf = memoryview(ptr).cast('B') 

        for y in range(h):
            row_off = y * bpl
            for x in range(w):
                i = row_off + x * 4
                b, g, r, a = buf[i], buf[i+1], buf[i+2], buf[i+3]
                if a == 255 and r >= thr and g >= thr and b >= thr:
                    buf[i+3] = 0

        return QtGui.QPixmap.fromImage(img)

    def _make_thumbnail_with_overlays(self, uuid_, sub_id=None, bbox=None, grouped=False, bg=None, border=None, target_size: int = 128) -> QtGui.QPixmap:
        # feat = self._load_feature_json(uuid_) or {}

        feat = self.thumb_json_cache.get(uuid_)
        if feat is None:
            feat = self._load_feature_json(uuid_) or {}
            self.thumb_json_cache[uuid_] = feat
        
        item_id = f"{uuid_}#sub_{sub_id}" if sub_id is not None else uuid_
        item = self.id2item.get(item_id) or self.id2item.get(uuid_)

        base = None

        if item is not None and getattr(item, "rgba", None) is not None:
            try:
                base = qpixmap_from_rgba(item.rgba)
            except Exception:
                base = None 

        if base is None or base.isNull():
            rel = feat.get("source_path")
            
            base = QtGui.QPixmap()
            
            if self.project_root and rel:
                if os.path.isabs(rel):
                    full_path = rel
                else:
                    full_path = os.path.join(self.project_root, rel)

                base = self.thumb_pixmap_cache.get(full_path)

                if base is None and os.path.exists(full_path):
                    base = QtGui.QPixmap(full_path)
                    if not base.isNull():
                        self.thumb_pixmap_cache[full_path] = base
                
                # if os.path.exists(full_path):
                #     base = QtGui.QPixmap(full_path)

        if base is None:
            base = QtGui.QPixmap()

        if (not isinstance(base, QtGui.QPixmap)) or base.isNull():
            base = QtGui.QPixmap(128, 128)
            base.fill(QtCore.Qt.darkGray)

        # try:
        #     base = self._apply_white_key(base)
        # except Exception as e:
        #     print(f"[WARN] _apply_white_key failed: {e}")

        pm = QtGui.QPixmap(target_size, target_size)
        pm.fill(QtCore.Qt.transparent)

        base_cache_key = (base.cacheKey(), target_size)
        scaled_base = self.thumb_scaled_base_cache.get(base_cache_key)

        if scaled_base is None:
            # 2. 如果快取沒有，才執行昂貴的 CPU 縮放
            original_size = base.size()
            q_target_size = QtCore.QSize(target_size, target_size)
            scaled_size = original_size.scaled(q_target_size, QtCore.Qt.KeepAspectRatio)
    
            scaled_base = base.scaled(scaled_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation) # <-- 瓶頸
            
            # 3. 存入快取
            self.thumb_scaled_base_cache.put(base_cache_key, scaled_base)
        
        # original_size = base.size()
        # q_target_size = QtCore.QSize(target_size, target_size)
        # scaled_size = original_size.scaled(q_target_size, QtCore.Qt.KeepAspectRatio)

        # scaled_base = base.scaled(scaled_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        if grouped and bg:
            painter.fillRect(pm.rect(), bg)

        x_offset = (pm.width() - scaled_base.width()) // 2
        y_offset = (pm.height() - scaled_base.height()) // 2

        painter.drawPixmap(x_offset, y_offset, scaled_base)
        
        if grouped and border:
            painter.setPen(QtGui.QPen(border, 3))
            painter.drawRect(pm.rect().adjusted(1, 1, -2, -2))

        # if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        #     x, y, w, h = bbox
        #     ow = base.width()
        #     oh = base.height()
            
        #     if ow > 0 and oh > 0:
        #         scale = min(pm.width() / float(ow), pm.height() / float(oh))
                
        #         rx = int(x * scale) + x_offset
        #         ry = int(y * scale) + y_offset
        #         rw = int(w * scale)
        #         rh = int(h * scale)
                
        #         painter.setPen(QtGui.QPen(QtGui.QColor(255, 128, 0, 230), 3))
        #         painter.drawRect(QtCore.QRect(rx, ry, rw, rh))

        painter.end()
        return pm

    
    def _on_select(self, uuid_, sub_id=None):
        self.current_selection = {"uuid": uuid_, "sub_id": sub_id}

        feat = self.features.load(uuid_) if self.features else None
        path_ok = bool(feat and feat.get("source_path"))
        self.btn_open_location.setEnabled(path_ok)

    def _refresh_file_list_grouped_mode(self):
        self._list_mode = "grouped"
        self.list_files.clear()
        self.list_files.setRowCount(0)
        self.list_files.setColumnCount(1)

        if not getattr(self, "object_groups", None):
            self._build_object_groups()
        groups = dict(self.object_groups or {})

        if not groups:
            try:
                if not getattr(self, "project_root", None):
                    raise RuntimeError("project_root is None")
                res_p = os.path.join(self.project_root, ".image_cache", "results.json")
                sim_groups = []
                if os.path.exists(res_p):
                    with open(res_p, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    sim_groups = res.get("similarity_groups") or []
                    if not sim_groups and hasattr(self, "group_view") and hasattr(self.group_view, "_build_groups_from_pairs"):
                        sim_groups = self.group_view._build_groups_from_pairs(res)

                for g in sim_groups:
                    gname = g.get("group_id") or "pairs"
                    for m in (g.get("members") or []):
                        groups.setdefault(gname, []).append({
                            "uuid": m.get("uuid"), "sub_id": m.get("sub_id"), "bbox": m.get("bbox")
                        })
            except Exception as e:
                print(f"[WARN] fallback to similarity_groups failed: {e}")

        if not groups:
            self.list_files.setRowCount(1)
            no_results_label = QtWidgets.QLabel("尚無分群結果")
            no_results_label.setAlignment(Qt.AlignCenter)
            self.list_files.setCellWidget(0, 0, no_results_label)
            self.list_files.setRowHeight(0, 100)
            return

        self.object_groups = groups

        self.list_files.setRowCount(len(groups))
        
        if self.dark_mode:
            colors = [QtGui.QColor("#111a2f"), QtGui.QColor("#0f162b")]
        else:
            colors = [QtGui.QColor("#eef2ff"), QtGui.QColor("#f7f8fb")]

        sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        icon_size = 120

        for row_index, (gname, members) in enumerate(sorted_groups):
            if not members:
                continue

            row_widget = QtWidgets.QWidget()
            row_widget.setAutoFillBackground(True)
            palette = row_widget.palette()
            palette.setColor(QtGui.QPalette.Window, colors[row_index % len(colors)])
            row_widget.setPalette(palette)
            
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(15, 8, 15, 8)
            row_layout.setSpacing(15)

            # --- 【修改點 1】---
            # (使用 row_index + 1 作為新的群組名稱)
            comp_name = f"comp_{row_index + 1}"
            header_label = QtWidgets.QLabel(f"<b>{comp_name}</b><br>({len(members)} 個)")
            # --- 【修改結束】---
            
            header_label.setFixedWidth(160)
            header_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            row_layout.addWidget(header_label)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll_area.setStyleSheet("QScrollArea, QWidget, QLabel { background: transparent; border: none; }")

            image_container = QtWidgets.QWidget()
            image_layout = QtWidgets.QHBoxLayout(image_container)
            image_layout.setContentsMargins(0, 0, 0, 0)
            image_layout.setSpacing(10)
            image_layout.setAlignment(Qt.AlignLeft)

            for m in members:
                uuid_ = m.get("uuid")
                sub_id = m.get("sub_id")
                bbox = m.get("bbox")
                
                pm = self._make_thumbnail_with_overlays(uuid_, sub_id, bbox, grouped=False, target_size=icon_size)
                
                # --- 【修改點 2】---
                # (metadata 中也使用新的 comp_name，確保點擊時傳遞正確)
                metadata = {"uuid": uuid_, "sub_id": sub_id, "group": comp_name, "bbox": bbox}
                # --- 【修改結束】---
                
                img_label = ImageLabel(metadata)
                img_label.setProperty("role", "thumb")
                img_label.setPixmap(pm)
                img_label.setFixedSize(icon_size, icon_size)
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setToolTip(f"UUID: {uuid_[:8]}\nSub ID: {sub_id}")
                
                img_label.clicked.connect(self._on_image_label_clicked)
                
                image_layout.addWidget(img_label)

            scroll_area.setWidget(image_container)
            row_layout.addWidget(scroll_area)

            self.list_files.setCellWidget(row_index, 0, row_widget)
            self.list_files.setRowHeight(row_index, icon_size + 20)

        self.lb_count.setText(f"群組：{len(groups)}")

    # def _refresh_file_list_grouped_mode(self):
    #     self._list_mode = "grouped"
    #     self.list_files.clear()
    #     self.list_files.setRowCount(0)
    #     self.list_files.setColumnCount(1)

    #     if not getattr(self, "object_groups", None):
    #         self._build_object_groups()
    #     groups = dict(self.object_groups or {})

    #     if not groups:
    #         try:
    #             if not getattr(self, "project_root", None):
    #                 raise RuntimeError("project_root is None")
    #             res_p = os.path.join(self.project_root, ".image_cache", "results.json")
    #             sim_groups = []
    #             if os.path.exists(res_p):
    #                 with open(res_p, "r", encoding="utf-8") as f:
    #                     res = json.load(f)
    #                 sim_groups = res.get("similarity_groups") or []
    #                 if not sim_groups and hasattr(self, "group_view") and hasattr(self.group_view, "_build_groups_from_pairs"):
    #                     sim_groups = self.group_view._build_groups_from_pairs(res)

    #             for g in sim_groups:
    #                 gname = g.get("group_id") or "pairs"
    #                 for m in (g.get("members") or []):
    #                     groups.setdefault(gname, []).append({
    #                         "uuid": m.get("uuid"), "sub_id": m.get("sub_id"), "bbox": m.get("bbox")
    #                     })
    #         except Exception as e:
    #             print(f"[WARN] fallback to similarity_groups failed: {e}")

    #     if not groups:
    #         self.list_files.setRowCount(1)
    #         no_results_label = QtWidgets.QLabel("尚無分群結果")
    #         no_results_label.setAlignment(Qt.AlignCenter)
    #         self.list_files.setCellWidget(0, 0, no_results_label)
    #         self.list_files.setRowHeight(0, 100)
    #         return

    #     self.object_groups = groups

    #     self.list_files.setRowCount(len(groups))
        
    #     if self.dark_mode:
    #         colors = [QtGui.QColor("#111a2f"), QtGui.QColor("#0f162b")]
    #     else:
    #         colors = [QtGui.QColor("#eef2ff"), QtGui.QColor("#f7f8fb")]

    #     sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    #     icon_size = 120

    #     for row_index, (gname, members) in enumerate(sorted_groups):
    #         if not members:
    #             continue

    #         row_widget = QtWidgets.QWidget()
    #         row_widget.setAutoFillBackground(True)
    #         palette = row_widget.palette()
    #         palette.setColor(QtGui.QPalette.Window, colors[row_index % len(colors)])
    #         row_widget.setPalette(palette)
            
    #         row_layout = QtWidgets.QHBoxLayout(row_widget)
    #         row_layout.setContentsMargins(15, 8, 15, 8)
    #         row_layout.setSpacing(15)

    #         header_label = QtWidgets.QLabel(f"<b>{gname}</b><br>({len(members)} 個)")
    #         header_label.setFixedWidth(160)
    #         header_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    #         row_layout.addWidget(header_label)

    #         scroll_area = QtWidgets.QScrollArea()
    #         scroll_area.setWidgetResizable(True)
    #         scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    #         scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    #         scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
    #         scroll_area.setStyleSheet("QScrollArea, QWidget, QLabel { background: transparent; border: none; }")

    #         image_container = QtWidgets.QWidget()
    #         image_layout = QtWidgets.QHBoxLayout(image_container)
    #         image_layout.setContentsMargins(0, 0, 0, 0)
    #         image_layout.setSpacing(10)
    #         image_layout.setAlignment(Qt.AlignLeft)

    #         for m in members:
    #             uuid_ = m.get("uuid")
    #             sub_id = m.get("sub_id")
    #             bbox = m.get("bbox")
                
    #             pm = self._make_thumbnail_with_overlays(uuid_, sub_id, bbox, grouped=False, target_size=icon_size)
                
    #             metadata = {"uuid": uuid_, "sub_id": sub_id, "group": gname, "bbox": bbox}
    #             img_label = ImageLabel(metadata)
    #             img_label.setProperty("role", "thumb")
    #             img_label.setPixmap(pm)
    #             img_label.setFixedSize(icon_size, icon_size)
    #             img_label.setAlignment(Qt.AlignCenter)
    #             img_label.setToolTip(f"UUID: {uuid_[:8]}\nSub ID: {sub_id}")
                
    #             img_label.clicked.connect(self._on_image_label_clicked)
                
    #             image_layout.addWidget(img_label)

    #         scroll_area.setWidget(image_container)
    #         row_layout.addWidget(scroll_area)

    #         self.list_files.setCellWidget(row_index, 0, row_widget)
    #         self.list_files.setRowHeight(row_index, icon_size + 20)

    #     self.lb_count.setText(f"群組：{len(groups)}")

    def _owner_key_and_pair(self, A, B):
        """回傳 (group_key, left_item, right_item)。
        規則：若有母圖→只掛『左邊那張母圖』；若左邊沒母圖而右邊有→交換左右；同母圖→掛該母圖；兩邊都沒母圖→散圖群。
        """
        pa = getattr(A, "parent_uuid", None)
        pb = getattr(B, "parent_uuid", None)

        if not pa and not pb:
            return SINGLES_GROUP_KEY, A, B

        if pa and pb and pa == pb:
            return pa, A, B

        if pa and not pb:
            return pa, A, B             
        if pb and not pa:
            return pb, B, A                

        return pa, A, B

    def highlight_pair(self, pair):
        A = pair["A"]; B = pair["B"]
        same_sheet = (A.get("sheet_id") == B.get("sheet_id")) and (A.get("sheet_id") is not None)

        if same_sheet:
            pix = self._pix_from_sheet_id(A["sheet_id"])
            if pix is not None:
                for v in (self.leftView, self.rightView):
                    v.show_image(pix)
                    sub_images = self._sub_images_of_sheet(A["sheet_id"]) or []
                    v.draw_bboxes(sub_images)
                self.leftView.highlight(A.get("sub_id"))
                if B.get("sub_id") in self.rightView._rect_items:
                    self.rightView._rect_items[B["sub_id"]].setPen(QtGui.QPen(self.rightView._pen_secondary))
        else:
            self.leftView.clear(); self.rightView.clear()
            if A.get("sheet_id"):
                pix = self._pix_from_sheet_id(A["sheet_id"])
                if pix is not None:
                    self.leftView.show_image(pix)
                    self.leftView.draw_bboxes(self._sub_images_of_sheet(A["sheet_id"]) or [])
                    self.leftView.highlight(A.get("sub_id"))
            elif A.get("rgba") is not None:
                self.leftView.show_image(qpixmap_from_rgba(A["rgba"]))

            if B.get("sheet_id"):
                pix = self._pix_from_sheet_id(B["sheet_id"])
                if pix is not None:
                    self.rightView.show_image(pix)
                    self.rightView.draw_bboxes(self._sub_images_of_sheet(B["sheet_id"]) or [])
                    sid = B.get("sub_id")
                    if sid:
                        for it in self.rightView._rect_items.values():
                            it.setPen(QtGui.QPen(self.rightView._pen_default))
                        if sid in self.rightView._rect_items:
                            self.rightView._rect_items[sid].setPen(QtGui.QPen(self.rightView._pen_secondary))
            elif B.get("rgba") is not None:
                self.rightView.show_image(qpixmap_from_rgba(B["rgba"]))

    def _load_sheet_rgba(self, parent_uuid):
        """取得 spritesheet 的 RGBA（先找 items_raw，找不到就從 features 的 source_path 讀檔）"""
        for it in self.items_raw:
            if getattr(it, "id", None) == parent_uuid:
                return it.rgba
        feat = getattr(self, "_load_feat", lambda _: None)(parent_uuid)
        if feat and feat.get("source_path") and getattr(self, "project_root", None):
            p = os.path.join(self.project_root, feat["source_path"])
            try:
                return read_image_rgba(p)
            except Exception:
                return None
        return None

    def _compose_side_by_side(self, left_rgba, right_rgba, gap=24, bg=(240, 242, 255, 255)):
        """把兩張圖水平排版到同一張畫布（回傳 RGBA）"""
        import numpy as np
        if left_rgba is None and right_rgba is None:
            return None
        if left_rgba is None:
            left_rgba = np.zeros((right_rgba.shape[0], 1, 4), dtype=np.uint8)
        if right_rgba is None:
            right_rgba = np.zeros((left_rgba.shape[0], 1, 4), dtype=np.uint8)

        h1, w1 = left_rgba.shape[:2]
        h2, w2 = right_rgba.shape[:2]
        H = max(h1, h2)
        W = w1 + gap + w2
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        canvas[:, :, 0] = bg[2]
        canvas[:, :, 1] = bg[1]
        canvas[:, :, 2] = bg[0]
        canvas[:, :, 3] = bg[3]

        y1 = (H - h1) // 2; y2 = (H - h2) // 2
        canvas[y1:y1+h1, 0:w1, :] = left_rgba
        x2 = w1 + gap
        canvas[y2:y2+h2, x2:x2+w2, :] = right_rgba
        return canvas

    def _draw_bbox_on_rgba(self, rgba, bbox, color=(255,0,0,255), t=4):
        """在 RGBA 上畫一個矩形框（不拉線段、直接畫 4 條邊）"""
        import numpy as np
        if rgba is None or bbox is None:
            return rgba
        x, y, w, h = map(int, bbox)
        H, W = rgba.shape[0], rgba.shape[1]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        rgba[y:y+t, x:x+w, :] = color
        rgba[y+h-t:y+h, x:x+w, :] = color

        rgba[y:y+h, x:x+t, :] = color
        rgba[y:y+h, x+w-t:x+w, :] = color
        return rgba

    def _decorate_panels(self):
        color = QtGui.QColor(0, 0, 0, 70)
        targets = [self.list_files]
        if hasattr(self, "group_view"):
            targets.append(self.group_view)

        for w in targets:
            eff = QtWidgets.QGraphicsDropShadowEffect(self)
            eff.setBlurRadius(20)
            eff.setOffset(0, 2)
            eff.setColor(color)
            w.setGraphicsEffect(eff)


    def _build_pairs_for_singles(self):
        singles = [it for it in self.items_raw if not getattr(it, "parent_uuid", None)]
        for it in singles:
            if getattr(it, "phash", None) is None:
                it.phash = phash_from_rgba(it.rgba)

        n = len(singles)
        for i in range(n):
            A = singles[i]
            for j in range(i + 1, n):
                B = singles[j]
                ham = _hamming64(A.phash, B.phash)
                if ham <= self.phash_hamming_max:
                    key = self._pair_key(A, B)
                    if key in self.seen_pair_keys:
                        continue
                    self.seen_pair_keys.add(key)
                    self.in_pair_ids.update([A.id, B.id])
                    self.pairs.append(PairHit(A.id, B.id, ham))

    def _build_pairs_cross_single_vs_sub(self):
        singles = [it for it in self.items_raw if not getattr(it, "parent_uuid", None)]
        subs    = [it for it in self.items_raw if     getattr(it, "parent_uuid", None)]

        for it in singles:
            if getattr(it, "phash", None) is None:
                it.phash = phash_from_rgba(it.rgba)
        for it in subs:
            if getattr(it, "phash", None) is None:
                it.phash = phash_from_rgba(it.rgba)

        for A in singles:
            for B in subs:
                ham = _hamming64(A.phash, B.phash)
                if ham <= self.phash_hamming_max:               
                    key = self._pair_key(A, B)
                    if key in self.seen_pair_keys:
                        continue
                    self.seen_pair_keys.add(key)
                    self.in_pair_ids.update([A.id, B.id])
                    self.pairs.append(PairHit(A.id, B.id, ham))

    @QtCore.pyqtSlot(str, str)
    def _open_pair_dialog_for_uuids(self, uuidA: str, uuidB: str):
        A = self.id2item.get(uuidA)
        B = self.id2item.get(uuidB)
        if not A or not B:
            QtWidgets.QMessageBox.information(self, "提示", "找不到配對影像（不在當前工作池）。")
            return
        ham = 0 
        dlg = PairDialog(A, B, ham, self)
        if dlg.exec_():
            choice = dlg.choice
            st = QtWidgets.QLabel()
            if getattr(self, "logger", None):
                self.logger.append("decision_made",
                    {"left": {"uuid": A.id, "name": A.display_name},
                    "right":{"uuid": B.id, "name": B.display_name}},
                    {"decision": choice})

    def _make_decision_cell(self, A, B, ham, status_label: QtWidgets.QLabel):
        btn = QtWidgets.QPushButton("決策")
        btn.setProperty("decided", False)
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(8, 4, 8, 4); lay.addStretch(1); lay.addWidget(btn); lay.addStretch(1)

        def handler():
            dlg = PairDialog(A, B, ham, self)
            if dlg.exec_():
                choice = dlg.choice
                if choice == "left":
                    A.keep, B.keep = True, False
                    status_label.setText("保留左"); status_label.setProperty("state", "keep")
                elif choice == "right":
                    A.keep, B.keep = False, True
                    status_label.setText("保留右"); status_label.setProperty("state", "keep")
                elif choice == "both":
                    A.keep, B.keep = True, True
                    status_label.setText("兩張都留"); status_label.setProperty("state", "both")
                else:
                    status_label.setText("略過");    status_label.setProperty("state", "skip")

                status_label.style().unpolish(status_label); status_label.style().polish(status_label); status_label.update()
                btn.setEnabled(False); btn.setText("已決策"); btn.setProperty("decided", True)
                btn.style().unpolish(btn); btn.style().polish(btn); btn.update()

                if getattr(self, "logger", None):
                    self.logger.append("decision_made",
                        {"left": {"uuid": A.id, "name": A.display_name},
                        "right":{"uuid": B.id, "name": B.display_name}},
                        {"decision": choice, "hamming": ham})

        btn.clicked.connect(handler)
        return wrap

    def _make_status_label(self, initial="未決策"):
        lab = QtWidgets.QLabel(initial)
        lab.setAlignment(Qt.AlignCenter)
        lab.setProperty("tag", "badge")
        lab.setProperty("state", "pending")
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(8, 4, 8, 4); lay.addStretch(1); lay.addWidget(lab); lay.addStretch(1)
        return wrap, lab

    def _pair_key(self, a, b):
        """無序配對 key：同一對 (A,B)/(B,A) 會得到相同 key。"""
        aid = getattr(a, "id", str(a))
        bid = getattr(b, "id", str(b))
        return (aid, bid) if aid < bid else (bid, aid)
    
    def _is_single_vs_sub(self, A, B) -> bool:
        """一邊是由 spritesheet 切出的子圖(src_path=None)，另一邊是原始單圖。"""
        return (getattr(A, "src_path", None) is None) ^ (getattr(B, "src_path", None) is None)

    
    def _finalize_intragroup_pairs(self, candidates):
        """
        candidates: list of (ham, A, B) 只包含同一張 spritesheet 內的候選
        以貪婪法做一對一配對（每個 id 只能被使用一次），避免一對多。
        """
        by_group = defaultdict(list)
        for ham, A, B in candidates:
            gid = getattr(A, "group_id", None)
            if gid is None:
                continue
            by_group[gid].append((ham, A, B))

        for gid, lst in by_group.items():
            used = set()
            for ham, A, B in sorted(lst, key=lambda t: t[0]):
                if A.id in used or B.id in used:
                    continue
                key = self._pair_key(A, B)
                if key in self.seen_pair_keys:
                    continue
                self.seen_pair_keys.add(key)
                self.in_pair_ids.update([A.id, B.id])
                self.pairs.append(PairHit(A.id, B.id, ham))
                used.add(A.id); used.add(B.id)

    def _ensure_group_node(self, key: str, mother_item: "ImageItem|None") -> QtWidgets.QTreeWidgetItem:
        node = self.group_nodes.get(key)
        if node:
            return node
        title = mother_item.display_name if mother_item else key
        node = QtWidgets.QTreeWidgetItem([title, "", "", "", ""])
        self.pair_tree.addTopLevelItem(node)
        node.setFirstColumnSpanned(True)
        node.setExpanded(True)
        self.group_nodes[key] = node
        return node

    def _add_pair_tree_row(self, group_node: QtWidgets.QTreeWidgetItem, A: ImageItem, B: ImageItem, ham: int):
        child = QtWidgets.QTreeWidgetItem(group_node, ["", "", str(ham), "", ""])
        iconL = make_thumb_rgba(A.rgba, max_w=64, max_h=64)
        iconR = make_thumb_rgba(B.rgba, max_w=64, max_h=64)
        child.setText(0, A.display_name); child.setIcon(0, iconL)
        child.setText(1, B.display_name); child.setIcon(1, iconR)

        st_wrap, st_label = self._make_status_label("未決策")
        self.pair_tree.setItemWidget(child, 4, st_wrap)
        dec_wrap = self._make_decision_cell(A, B, ham, st_label)
        self.pair_tree.setItemWidget(child, 3, dec_wrap)

        child.setData(0, Qt.UserRole, {"left": A.id, "right": B.id, "hamming": ham})

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.act_add = QtWidgets.QAction(QtGui.QIcon.fromTheme("list-add"), "新增圖片", self)

        self.act_add_dir = QtWidgets.QAction(QtGui.QIcon.fromTheme("folder-open"), "新增資料夾", self)
        self.act_add_dir.setShortcut("Ctrl+Shift+O")

        self.act_clear = QtWidgets.QAction(QtGui.QIcon.fromTheme("edit-clear"), "清空", self)
        self.act_params = QtWidgets.QAction(QtGui.QIcon.fromTheme("preferences-system"), "參數…", self)
        self.act_params.setVisible(False)
        self.act_run = QtWidgets.QAction(QtGui.QIcon.fromTheme("system-run"), "開始處理", self)
        self.act_theme = QtWidgets.QAction("🌗 主題", self)
        self.act_theme.setCheckable(True)

        for a in (self.act_add, self.act_add_dir, self.act_clear, self.act_theme, self.act_params, self.act_run):
            tb.addAction(a)

        self.act_add.triggered.connect(self.on_add)
        self.act_add_dir.triggered.connect(self.on_add_dir)

        self.act_clear.triggered.connect(self.on_clear)
        self.act_params.triggered.connect(self.on_params)
        self.act_run.triggered.connect(self.on_run)

        # act_add = QtWidgets.QAction(QtGui.QIcon.fromTheme("list-add"), "新增圖片", self)

        # act_add_dir = QtWidgets.QAction(QtGui.QIcon.fromTheme("folder-open"), "新增資料夾", self)
        # act_add_dir.setShortcut("Ctrl+Shift+O")

        # act_clear = QtWidgets.QAction(QtGui.QIcon.fromTheme("edit-clear"), "清空", self)
        # act_params = QtWidgets.QAction(QtGui.QIcon.fromTheme("preferences-system"), "參數…", self)
        # act_params.setVisible(False)
        # act_run = QtWidgets.QAction(QtGui.QIcon.fromTheme("system-run"), "開始處理", self)
        # act_run.setObjectName("act_run")
        # self.act_theme = QtWidgets.QAction("🌗 主題", self)
        # self.act_theme.setCheckable(True)

        # for a in (act_add, act_add_dir, act_clear, self.act_theme, act_params, act_run):
        #     tb.addAction(a)

        # act_add.triggered.connect(self.on_add)
        # act_add_dir.triggered.connect(self.on_add_dir)

        # act_clear.triggered.connect(self.on_clear)
        # act_params.triggered.connect(self.on_params)
        # act_run.triggered.connect(self.on_run)
        self.act_theme.triggered.connect(self.toggle_theme)

        self.act_add.setShortcut("Ctrl+O")
        self.act_run.setShortcut("Ctrl+R")
        self.act_clear.setShortcut("Ctrl+Backspace")

    def _build_central(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left_panel = QtWidgets.QWidget()
        lyt = QtWidgets.QVBoxLayout(left_panel)
        top_row = QtWidgets.QHBoxLayout()
        self.lb_count = QtWidgets.QLabel("已加入：0")
        self.lb_count.setProperty("tag", "badge")
        top_row.addWidget(self.lb_count)
        top_row.addStretch(1)
        lyt.addLayout(top_row)

        self.list_files = QtWidgets.QTableWidget()
        self.list_files.setObjectName("fileList")
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_files.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_files.setShowGrid(False)
        self.list_files.verticalHeader().setVisible(False)
        self.list_files.horizontalHeader().setVisible(False)
        self.list_files.setColumnCount(1)
        self.list_files.horizontalHeader().setStretchLastSection(True)
        lyt.addWidget(self.list_files, 1)

        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        rlyt = QtWidgets.QVBoxLayout(right_panel)
        rlyt.setContentsMargins(0, 0, 0, 0)

        rlyt.addWidget(self.group_view, 1)

        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        info_widget = self.group_view._build_info_panel()
        rlyt.addWidget(info_widget, 0)

        self.group_view.attach_external_info_panel(
            info_widget,
            (
                self.group_view.info_uuid,
                self.group_view.info_subid,
                self.group_view.info_size,
                self.group_view.info_source,
                self.group_view.info_path,
                self.group_view.info_count,
            ),
        )
        info_widget.setObjectName("infoPanel")

    def _reveal_current_file(self):
        if hasattr(self, "_open_current_in_explorer"):
            return self._open_current_in_explorer()
        return

    def _on_left_file_clicked(self, lw_item: QtWidgets.QListWidgetItem):
        """左側點一張 → 右側群組樹自動選到同一張（member 列）"""
        data = lw_item.data(QtCore.Qt.UserRole) or {}
        uuid_ = data.get("uuid")

        if not uuid_:
            p = lw_item.toolTip() or lw_item.text()
            for it in self.items_raw:
                if it.src_path and (
                    it.src_path == p or
                    (self.project_root and os.path.relpath(it.src_path, self.project_root) == p) or
                    os.path.basename(it.src_path) == os.path.basename(p)
                ):
                    uuid_ = it.id
                    break

        if uuid_ and hasattr(self, "group_view") and self.group_view:
            self.group_view.select_member_by_uuid(uuid_)

    def _mother_pixmap(self, parent_uuid: str):
        if not parent_uuid or not getattr(self, "project_root", None):
            return None
        mf = None
        if hasattr(self, "group_view") and hasattr(self.group_view, "_load_feat"):
            mf = self.group_view._load_feat(parent_uuid)
        if not mf or not mf.get("source_path"):
            return None
        p = os.path.join(self.project_root, mf["source_path"])
        pm = QtGui.QPixmap(p)
        return pm if not pm.isNull() else None

    def _crop_from_sheet(self, parent_uuid: str, bbox):
        pm = self._mother_pixmap(parent_uuid)
        if not pm or not bbox:
            return None
        x, y, w, h = [int(v) for v in bbox]
        r = QtCore.QRect(x, y, w, h).intersected(pm.rect())
        if r.isEmpty():
            return None
        return pm.copy(r)

    def _load_feat(self, uuid_: str):
        return self._load_feature_json(uuid_)

    def _group_key_for_pair(self, A, B):
        pa = getattr(A, "parent_uuid", None)
        pb = getattr(B, "parent_uuid", None)

        if pa and pb and pa == pb:
            return pa
        if pa and pb:
            return pb
        if pa:
            return pa
        if pb:
            return pb
        return SINGLES_GROUP_KEY

    def _rebuild_pair_groups(self):
        self.pair_tree.clear()
        self.leftView.clear()
        self.rightView.clear()
        if not self.pairs:
            return

        groups = {}
        for parent_uuid, meta in getattr(self, "sheet_meta", {}).items():
            groups[parent_uuid] = {"mother": self._load_feat(parent_uuid), "pairs": []}
        groups[SINGLES_GROUP_KEY] = {"mother": None, "pairs": []}

        for ph in self.pairs:
            A = self.id2item.get(ph.left_id); B = self.id2item.get(ph.right_id)
            if not A or not B:
                continue

            pa = getattr(A, "parent_uuid", None)
            pb = getattr(B, "parent_uuid", None)

            def _ensure(gk):
                if gk not in groups:
                    groups[gk] = {"mother": (None if gk == SINGLES_GROUP_KEY else self._load_feat(gk)), "pairs": []}

            if pa and pb and pa != pb:
                _ensure(pa); _ensure(pb)
                left_pa, right_pa = (A, B) if getattr(A, "parent_uuid", None) == pa else (B, A)
                groups[pa]["pairs"].append((left_pa, right_pa, ph.hamming))
                left_pb, right_pb = (A, B) if getattr(A, "parent_uuid", None) == pb else (B, A)
                groups[pb]["pairs"].append((left_pb, right_pb, ph.hamming))
            else:
                gkey, left_it, right_it = self._owner_key_and_pair(A, B)
                _ensure(gkey)
                groups[gkey]["pairs"].append((left_it, right_it, ph.hamming))

        for parent_uuid, data in groups.items():
            if parent_uuid == SINGLES_GROUP_KEY:
                top = QtWidgets.QTreeWidgetItem(self.pair_tree, ["散圖", "-", "-", "-", "-"])
            else:
                mf = data["mother"] or {}
                title = mf.get("source_path") or f"sheet_{parent_uuid[:8]}"
                top = QtWidgets.QTreeWidgetItem(self.pair_tree, [f"Spritesheet: {title}", f"{parent_uuid[:8]}", "-", "-", "-"])

            top.setData(
                0, QtCore.Qt.UserRole,
                {"type": "group", "parent_uuid": parent_uuid, "group_id": parent_uuid}
            )


            for A, B, ham in data["pairs"]:
                child = QtWidgets.QTreeWidgetItem(top, ["配對", A.display_name, B.display_name, "", "未決策"])
                child.setData(0, QtCore.Qt.UserRole, {"type": "pair", "parent_uuid": parent_uuid,
                                                    "left_id": A.id, "right_id": B.id})
                btn = QtWidgets.QPushButton("決策")
                btn.clicked.connect(lambda _, a=A, b=B, item=child: self._on_decide_pair_item(a, b, item))
                self.pair_tree.setItemWidget(child, 3, btn)

            top.setExpanded(True)

        if self.pair_tree.topLevelItemCount() > 0:
            self.pair_tree.setCurrentItem(self.pair_tree.topLevelItem(0))
            self._on_pair_tree_select()

    def _show_one_side_sheet(self, view: BBoxGraphicsView, parent_uuid):
        view.clear()
        if not parent_uuid: return
        pm = self._mother_pixmap(parent_uuid)
        if not pm: return
        view.show_image(pm, fit=False)
        mf = self._load_feat(parent_uuid) or {}
        view.draw_bboxes(mf.get("sub_images") or [])

    def _on_pair_tree_select(self):
        """群組樹／成員被點選時：右側顯示影像 + 更新資訊欄。"""
        it = self.tree.currentItem() if hasattr(self, "tree") else None
        self._last_selected_meta = None
        self.rightView.clear()
        if not it:
            self._update_info_panel(None, None, None)
            return

        meta = it.data(0, QtCore.Qt.UserRole) or {}
        self._last_selected_meta = meta

        if meta.get("type") == "group":
            gid = meta.get("group_id") or meta.get("parent_uuid")
            self._update_info_panel(None, None, gid)
            return
        
        uuid_  = meta.get("uuid")
        sub_id = meta.get("sub_id")
        gid    = meta.get("group_id")
        self.current_uuid = uuid_

        if sub_id is not None:
            parent_uuid = uuid_ 
            mother = self._load_feat(parent_uuid) or {}
            rel = mother.get("source_path")
            
            if rel and self.project_root:
                pm = self._mother_pixmap(parent_uuid)
                if pm:
                    self.rightView.show_image(pm, fit=True)
                    
                    all_bboxes = (mother.get("sub_images") or [])
                    target_bbox = None
                    
                    id_to_find = sub_id 

                    for si in all_bboxes:
                        if si.get("sub_id") == id_to_find:
                            target_bbox = si
                            break
                    
                    if target_bbox:
                        original_sid_str = str(id_to_find)
                        
                        self.rightView.draw_bboxes([target_bbox]) #
                        self.rightView.focus_bbox(original_sid_str) #
            
            self._update_info_panel(uuid_, sub_id, gid)
            if hasattr(self, "btn_open_folder"):
                self.btn_open_folder.setEnabled(bool(rel))
            return

        feat = self._load_feat(uuid_) or {} 
        rel = feat.get("source_path")
        if rel and self.project_root:
            abs_p = os.path.join(self.project_root, rel)
            pm = QtGui.QPixmap(abs_p)
            if not pm.isNull():
                self.rightView.show_image(pm, fit=True)
        
        self._update_info_panel(uuid_, None, gid)
        if hasattr(self, "btn_open_folder"):
            self.btn_open_folder.setEnabled(bool(rel))

    def _on_decide_pair_item(self, A, B, item):
        ham = 0
        dlg = PairDialog(A, B, ham, self)
        if dlg.exec_():
            choice = dlg.choice
            if choice == "keep_left":
                status_text = "保留左"
                state = "keep"
            elif choice == "keep_right":
                status_text = "保留右"
                state = "keep"
            elif choice == "keep_both":
                status_text = "兩張都留"
                state = "both"
            else:
                status_text = "已決策"
                state = "skip"

            lbl = QtWidgets.QLabel(status_text)
            lbl.setProperty("tag", "badge")
            lbl.setProperty("state", state)
            lbl.setAlignment(QtCore.Qt.AlignCenter)

            self.pair_tree.setItemWidget(item, 4, lbl)

            w = self.pair_tree.itemWidget(item, 3)
            if w:
                w.setText("已決策")
                w.setEnabled(False)

            if getattr(self, "logger", None):
                self.logger.append(
                    "decision_made",
                    {"left": {"uuid": A.id, "name": A.display_name},
                    "right": {"uuid": B.id, "name": B.display_name}},
                    {"decision": choice}
                )

            self._on_pair_tree_select()
            QtWidgets.QApplication.processEvents()

            meta = item.data(0, QtCore.Qt.UserRole) or {}
            p_uuid = meta.get("parent_uuid")

            left_sid  = A.sub_id if getattr(A, "parent_uuid", None) == p_uuid and getattr(A, "sub_id", None) else None
            right_sid = B.sub_id if getattr(B, "parent_uuid", None) == p_uuid and getattr(B, "sub_id", None) else None

            if left_sid and hasattr(self.leftView, "focus_bbox"):
                self.leftView.focus_bbox(left_sid, margin=16, use_secondary=False)
            if right_sid and hasattr(self.rightView, "focus_bbox"):
                self.rightView.focus_bbox(right_sid, margin=16, use_secondary=True)

    def _build_statusbar(self):
        sb = QtWidgets.QStatusBar()
        self.setStatusBar(sb)

        self.sb_text = QtWidgets.QLabel("就緒")
        sb.addWidget(self.sb_text)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        sb.addPermanentWidget(self.progress, 1)

        self.lb_pairs = QtWidgets.QLabel("相似結果：0 組（0 群）")
        sb.addPermanentWidget(self.lb_pairs)

    def _apply_theme(self):
        self.setStyleSheet(DARK_QSS if self.dark_mode else LIGHT_QSS)
        self.act_theme.setChecked(self.dark_mode)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _image_exts(self) -> set[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _iter_image_files(self, root_dir: str):
        exts = self._image_exts()
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    yield os.path.join(dirpath, fn)

    def _index_is_clean_by_uuid(self, uuid_: str) -> bool:
        try:
            if not getattr(self, "index", None):
                return False
            imap = (self.index.data or {}).get("image_map", {})
            for _, meta in imap.items():
                if meta.get("uuid") == uuid_:
                    return not meta.get("dirty_features", True)
        except Exception:
            pass
        return False

    def _load_cached_features_for_item(self, it) -> dict | None:
        """
        單張：直接讀 {uuid}.json 的 features
        子圖：讀母圖 {parent_uuid}.json -> sub_images[].features
        """
        try:
            if not getattr(self, "features", None):
                return None
            if getattr(it, "parent_uuid", None) is None:
                feat = self.features.load(it.id)
                return (feat or {}).get("features")
            else:
                mother = self.features.load(it.parent_uuid)
                if not mother:
                    return None
                for si in (mother.get("sub_images") or []):
                    sid = si.get("sub_id")
                    if isinstance(sid, str) and sid.startswith("sub_"):
                        try:
                            sid = int(sid.split("_")[1])
                        except Exception:
                            pass
                    if sid == it.sub_id:
                        return (si.get("features") or {})
        except Exception:
            return None
        return None

    def on_add_dir(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if not root: return
        self.project_root = root
        os.makedirs(os.path.join(root, ".image_cache"), exist_ok=True)
        self.index = IndexStore(root)
        self.features = FeatureStore(root)
        self.logger = ActionsLogger(root)
        self.logger.append("scan_started", {"project_root": root}, {"include_exts": list(self._image_exts())})
        self.group_view.set_project_root(self.project_root)

        existing_paths = {it.src_path for it in self.items_raw if it.src_path}
        added = errors = 0
        
        all_files = list(self._iter_image_files(root))
        
        for p in all_files:
            abs_p = os.path.abspath(p)

            if abs_p in self._input_paths:
                continue
            
            try:
                rel = os.path.relpath(abs_p, root)
                uid = self.index.touch_file(abs_p)   
                
                self._input_paths.add(abs_p)      
                self._input_order.append((uid, abs_p))
                added += 1
            except Exception as e:
                errors += 1
        
        self.index.save()
        self.logger.append("scan_finished", {"project_root": root}, {"new_or_touched": added, "errors": errors})
        
        self._refresh_file_list_input_mode()

        if added or errors:
            msg = f"從資料夾加入 {added} 張圖片"
            if errors:
                msg += f"（失敗 {errors}）"
            self.sb_text.setText(msg)
        else:
            self.sb_text.setText("資料夾內未找到可用圖片")

    # def on_add_dir(self):
    #     root = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇資料夾")
    #     if not root: return
    #     self.project_root = root
    #     os.makedirs(os.path.join(root, ".image_cache"), exist_ok=True)
    #     self.index = IndexStore(root)
    #     self.features = FeatureStore(root)
    #     self.logger = ActionsLogger(root)
    #     self.logger.append("scan_started", {"project_root": root}, {"include_exts": list(self._image_exts())})
    #     self.group_view.set_project_root(self.project_root)

    #     existing_paths = {it.src_path for it in self.items_raw if it.src_path}
    #     added = errors = 0
        
    #     all_files = list(self._iter_image_files(root))
        
    #     for p in all_files:
    #         if p in self._input_paths: # <-- 【修改】使用 self._input_paths
    #             continue
    #         try:
    #             rel = os.path.relpath(p, root)
    #             uid = self.index.touch_file(p)
    #             self._input_paths.add(p)
    #             self._input_order.append((uid, p))
    #             added += 1
    #         except Exception as e:
    #             errors += 1
        
    #     self.index.save()
    #     self.logger.append("scan_finished", {"project_root": root}, {"new_or_touched": added, "errors": errors})
        
    #     self._refresh_file_list_input_mode()

    #     if added or errors:
    #         msg = f"從資料夾加入 {added} 張圖片"
    #         if errors:
    #             msg += f"（失敗 {errors}）"
    #         self.sb_text.setText(msg)
    #     else:
    #         self.sb_text.setText("資料夾內未找到可用圖片")

    def on_add(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")

        if files and not self.project_root:
            root = os.path.dirname(files[0])
            self.project_root = root
            os.makedirs(os.path.join(root, ".image_cache"), exist_ok=True)
            self.index = IndexStore(root)
            self.features = FeatureStore(root)
            self.logger = ActionsLogger(root)
            self.logger.append("scan_started", {"project_root": root}, {"include_exts": list(self._image_exts())})
            self.group_view.set_project_root(self.project_root)

        added = 0
        for p in files:
            abs_p = os.path.abspath(p)

            if abs_p in self._input_paths: 
                continue

            try:
                uid = self.index.touch_file(abs_p) if self.index else str(uuid.uuid4())
                
                self._input_paths.add(abs_p)
                self._input_order.append((uid, abs_p))
                added += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "讀檔失敗", f"{p}\n{e}")

        if self.index:
            self.index.save()
            self.logger.append("scan_finished", {"project_root": self.project_root}, {"new_or_touched": added, "errors": 0})

        self._refresh_file_list_input_mode()
        if added:
            self.sb_text.setText(f"新增 {added} 張圖片")

    # def on_add(self):
    #     files, _ = QtWidgets.QFileDialog.getOpenFileNames(
    #         self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")

    #     if files and not self.project_root:
    #         root = os.path.dirname(files[0])
    #         self.project_root = root
    #         os.makedirs(os.path.join(root, ".image_cache"), exist_ok=True)
    #         self.index = IndexStore(root)
    #         self.features = FeatureStore(root)
    #         self.logger = ActionsLogger(root)
    #         self.logger.append("scan_started", {"project_root": root}, {"include_exts": list(self._image_exts())})
    #         self.group_view.set_project_root(self.project_root)

    #     added = 0
    #     for p in files:
    #         abs_p = os.path.abspath(p)

    #         if abs_p in self._input_paths: 
    #             continue

    #         try:
    #             uid = self.index.touch_file(abs_p) if self.index else str(uuid.uuid4())
                
    #             self._input_paths.add(abs_p)
    #             self._input_order.append((uid, abs_p))
    #             added += 1
    #         except Exception as e:
    #             QtWidgets.QMessageBox.warning(self, "讀檔失敗", f"{p}\n{e}")

    #     if self.index:
    #         self.index.save()
    #         self.logger.append("scan_finished", {"project_root": self.project_root}, {"new_or_touched": added, "errors": 0})

    #     self._refresh_file_list_input_mode()
    #     if added:
    #         self.sb_text.setText(f"新增 {added} 張圖片")

    def on_clear(self):
        self.items_raw.clear()
        self.pool.clear()
        self.id2item.clear()
        self.pairs.clear()
        self.in_pair_ids.clear()
        self.seen_pair_keys.clear()
        self._input_order.clear()
        self._input_paths.clear()

        self.thumb_json_cache.clear()
        self.thumb_pixmap_cache.clear()
        self.thumb_scaled_base_cache = LRUCache(capacity=50)

        self.list_files.clear()
        self.list_files.setRowCount(0)
        self.lb_count.setText("已加入：0")
        self.lb_pairs.setText("相似結果：0 組")
        self.sb_text.setText("已清空")

        if hasattr(self, "group_view"):
            try:
                self.group_view.tree.clear()
                self.group_view.leftView.clear()
                self.group_view.rightView.clear()
                self.group_view._update_info_panel(None, None, None)
            except Exception:
                pass

        self.progress.setRange(0, 0) 
        self.progress.setValue(0)

        self.act_run.setEnabled(True)
        self.act_add.setEnabled(True)
        self.act_add_dir.setEnabled(True)
        self.act_clear.setEnabled(True)

    def on_params(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("參數設定")
        lay = QtWidgets.QFormLayout(dlg)

        e_phash = QtWidgets.QSpinBox(); e_phash.setRange(0, 64); e_phash.setValue(self.phash_hamming_max)
        e_alpha = QtWidgets.QSpinBox(); e_alpha.setRange(0, 255); e_alpha.setValue(self.alpha_thr)
        e_area  = QtWidgets.QSpinBox(); e_area.setRange(0, 500000); e_area.setValue(self.min_area)
        e_size  = QtWidgets.QSpinBox(); e_size.setRange(1, 2000); e_size.setValue(self.min_size)
        e_seg   = QtWidgets.QSpinBox(); e_seg.setRange(1, 999); e_seg.setValue(self.spr_min_segs)
        e_cov   = QtWidgets.QDoubleSpinBox(); e_cov.setRange(0.0, 1.0); e_cov.setSingleStep(0.01); e_cov.setValue(self.spr_min_cover)

        lay.addRow("pHash Hamming 門檻（相似 ≤）", e_phash)
        lay.addRow("Alpha-CC α 門檻", e_alpha)
        lay.addRow("最小面積", e_area)
        lay.addRow("最小邊長", e_size)
        lay.addRow("組圖判定最少片段數", e_seg)
        lay.addRow("組圖判定覆蓋率下限", e_cov)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        lay.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.phash_hamming_max = e_phash.value()
            self.alpha_thr = e_alpha.value()
            self.min_area = e_area.value()
            self.min_size = e_size.value()
            self.spr_min_segs = e_seg.value()
            self.spr_min_cover = e_cov.value()
            self.sb_text.setText("已更新參數")

    # app/ui/main_window.py -> class MainWindow

    def on_run(self):
        if self.worker_thread is not None:
            QtWidgets.QMessageBox.information(self, "提示", "正在處理中，請稍候...")
            return

        if not self._input_order:
            QtWidgets.QMessageBox.information(self, "提示", "請先新增圖片。")
            return
        
        # 1. 禁用按鈕，重設進度條
        try:
            # (您需要確保 "開始處理" 的 QAction 有一個 objectName)
            # (或者在 _build_toolbar 時儲存 self.act_run)
            # self.findChild(QtWidgets.QAction, "act_run").setEnabled(False)
            self.act_run.setEnabled(False)
            self.act_add.setEnabled(False)
            self.act_add_dir.setEnabled(False)
            self.act_clear.setEnabled(False)
        except Exception:
            print("Warning: Cound not find act_run to disable.")
            
        self.sb_text.setText("正在準備...")
        self.progress.setRange(0, 0) # 進入忙碌狀態
        self.progress.setValue(0)
        self.items_raw.clear() # 清理舊資料
        self.id2item.clear()
        self.pool.clear()
        self.pairs.clear()

        # 2. 打包所有需要的參數
        task_args = {
            "input_order": list(self._input_order), # 傳遞副本
            "project_root": self.project_root,
            "alpha_thr": self.alpha_thr,
            "min_area": self.min_area,
            "min_size": self.min_size,
            "spr_min_segs": self.spr_min_segs,
            "spr_min_cover": self.spr_min_cover,
            "phash_hamming_max_intra": self.phash_hamming_max_intra,
            "phash_hamming_max": self.phash_hamming_max,
        }

        # 3. 建立 Thread 和 Worker
        self.worker_thread = QtCore.QThread()
        self.worker = ScanWorker(task_args)
        self.worker.moveToThread(self.worker_thread)

        # 4. 連接信號
        self.worker.progressInit.connect(self.progress.setRange)
        self.worker.progressStep.connect(self.progress.setValue)
        self.worker.logMessage.connect(self.sb_text.setText)
        self.worker.finished.connect(self.on_run_finished) # <-- 新增的接收器
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_worker_thread_finished) # <-- 新增的清理器

        # 5. 啟動
        self.worker_thread.start()

    @QtCore.pyqtSlot()
    def _on_worker_thread_finished(self):
        """(私有) QThread 結束時，清理參照並重新啟用按鈕"""
        self.worker_thread = None
        self.worker = None
        try:
            # self.findChild(QtWidgets.QAction, "act_run").setEnabled(True)
            self.act_run.setEnabled(False)
            self.act_add.setEnabled(False)
            self.act_add_dir.setEnabled(False)
            self.act_clear.setEnabled(True)
        except Exception:
            print("Warning: Could not find act_run to enable.")

    @QtCore.pyqtSlot(dict)
    def on_run_finished(self, results):
        """(Slot) 接收 Worker 執行完畢的訊號"""
        
        # 1. 重設 UI
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # 2. 檢查錯誤
        if results.get("error"):
            self.sb_text.setText(f"處理失敗: {results['error']}")
            QtWidgets.QMessageBox.critical(self, "處理失敗", results['error'])
            return

        # 3. 將結果倒回 MainWindow
        self.pairs = results.get("pairs", [])
        self.id2item = results.get("id2item", {})
        self.pool = results.get("pool", [])
        self.items_raw = results.get("items_raw", [])
        self.in_pair_ids = results.get("in_pair_ids", set())
        self.seen_pair_keys = results.get("seen_pair_keys", set())

        # --- 【 關鍵修正 】 ---
        # 1. 清除舊快取
        self.thumb_json_cache.clear()
        self.thumb_pixmap_cache.clear()
        self.thumb_scaled_base_cache = LRUCache(capacity=50)

        # 2. (新) 從 Worker 取得記憶體中的 JSON Payloads
        self.all_json_payloads = results.get("json_payloads", {})

        # 3. (新) 將 Payloads 預先載入 JSON 快取中
        if self.all_json_payloads:
            for uuid_, payload in self.all_json_payloads.items():
                self.thumb_json_cache[uuid_] = payload
        # --- 【 修正結束 】 ---

        # 4. 更新 UI (載入 group_view, 刷新左側列表)
        if getattr(self, "project_root", None):
            try:
                # group_view 讀取 worker 剛才寫入的 results.json
                self.group_view.load_from_results()
            except Exception as e:
                self.sb_text.setText(f"載入結果時出錯: {e}")
                print(f"[ERROR] load_from_results failed: {e}")

        self.progress.setValue(100)
        self.lb_pairs.setText(f"相似結果：{len(self.pairs)} 組")
        self.sb_text.setText("配對完成，請於下方查看群組結果")
        
        # 5. 刷新左側列表為「群組模式」
        # (現在這個呼叫會 100% 命中快取，所以很快)
        self._refresh_file_list_grouped_mode()
        self.progress.setValue(100)

    # @QtCore.pyqtSlot(dict)
    # def on_run_finished(self, results):
    #     """(Slot) 接收 Worker 執行完畢的訊號"""
        
    #     # 1. 重設 UI
    #     self.progress.setRange(0, 100)
    #     self.progress.setValue(0)

    #     # 2. 檢查錯誤
    #     if results.get("error"):
    #         self.sb_text.setText(f"處理失敗: {results['error']}")
    #         QtWidgets.QMessageBox.critical(self, "處理失敗", results['error'])
    #         return

    #     # 3. 將結果倒回 MainWindow
    #     self.pairs = results.get("pairs", [])
    #     self.id2item = results.get("id2item", {})
    #     self.pool = results.get("pool", [])
    #     self.items_raw = results.get("items_raw", [])
    #     self.in_pair_ids = results.get("in_pair_ids", set())
    #     self.seen_pair_keys = results.get("seen_pair_keys", set())

    #     self.thumb_json_cache.clear()
    #     self.thumb_pixmap_cache.clear()
    #     self.thumb_scaled_base_cache = LRUCache(capacity=50)

    #     features_data = results.get("features_data", {})

    #     # 4. 更新 UI (載入 group_view, 刷新左側列表)
    #     if getattr(self, "project_root", None):
    #         try:
    #             # group_view 讀取 worker 剛才寫入的 results.json
    #             self.group_view.load_from_results()
    #         except Exception as e:
    #             self.sb_text.setText(f"載入結果時出錯: {e}")
    #             print(f"[ERROR] load_from_results failed: {e}")

    #     self.progress.setValue(100)
    #     self.lb_pairs.setText(f"相似結果：{len(self.pairs)} 組")
    #     self.sb_text.setText("配對完成，請於下方查看群組結果")
        
    #     # 5. 刷新左側列表為「群組模式」
    #     self._refresh_file_list_grouped_mode(features_data)
    #     # self._refresh_file_list_grouped_mode()
    #     self.progress.setValue(100)

    def on_export(self):
        if not self.pool:
            QtWidgets.QMessageBox.information(self, "提示", "沒有可匯出的資料。")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if not out_dir:
            return

        self.logger and self.logger.append("export_started", {"destination": out_dir}, {})

        in_pairs = getattr(self, "in_pair_ids", set())

        undecided = [it for it in self.pool if (it.id in in_pairs) and (it.keep is None)]
        if undecided:
            QtWidgets.QMessageBox.warning(
                self, "尚有未決策",
                f"還有 {len(undecided)} 筆相似配對尚未決策，請先完成決策再匯出。"
            )
            return

        def _unique_path(dst: str) -> str:
            base, ext = os.path.splitext(dst)
            k = 1
            while os.path.exists(dst):
                dst = f"{base}_{k}{ext}"
                k += 1
            return dst

        exported = 0
        seen = set()

        for it in self.pool:
            if it.id in seen:
                continue
            seen.add(it.id)

            if it.id in in_pairs:
                if it.keep is not True:
                    continue

            if it.src_path is None:
                safe_name = it.display_name.replace("/", "_").replace("\\", "_")
                if not safe_name.lower().endswith(".png"):
                    safe_name += ".png"
                dst = os.path.join(out_dir, safe_name)
                dst = _unique_path(dst)
                Image.fromarray(it.rgba).save(dst)
                exported += 1
            else:
                base = os.path.basename(it.src_path)
                dst = os.path.join(out_dir, base)
                dst = _unique_path(dst)
                shutil.copy2(it.src_path, dst)
                exported += 1

        self.logger and self.logger.append("export_finished", {"destination": out_dir},
                                       {"exported": int(exported)})
        
        QtWidgets.QMessageBox.information(self, "匯出完成", f"已匯出 {exported} 個檔案到：\n{out_dir}")

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(e)