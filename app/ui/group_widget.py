import os, sys, json, subprocess
from typing import Optional, Tuple
from PyQt5 import QtGui, QtCore, QtWidgets
from collections import defaultdict, OrderedDict
from ..constants import (
    SHAPE_ALPHA_THR, PHASH_SHAPE_MAX, ASPECT_TOL, CANON_PAD_SECONDARY, SINGLES_GROUP_KEY, SINGLES_BUCKET_SHIFT
)
from ..core.phash import (
    phash_from_canon_alpha, phash_from_canon_rgba, best_rot_hamming_fast
)
from ..core.features import crop_aspect_ratio
from .widgets import BBoxGraphicsView
from .dialogs import PairDecisionDialog as PairDialog

class WorkerSignals(QtCore.QObject):
    loaded = QtCore.pyqtSignal(str, object)

class LoaderRunnable(QtCore.QRunnable):
    def __init__(self, path, uuid_, signal_emitter):
        super().__init__()
        self.path = path
        self.uuid = uuid_
        self.emitter = signal_emitter

    @QtCore.pyqtSlot()
    def run(self):
        if self.path and os.path.exists(self.path):
            img = QtGui.QImage(self.path)
            self.emitter.loaded.emit(self.uuid, img)
        else:
            self.emitter.loaded.emit(self.uuid, QtGui.QImage())

class LRUCache:
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

def _get_sig(obj: dict | None) -> str | None:
    if not isinstance(obj, dict):
        return None
    src = obj.get("features", obj)
    sig = src.get("signature")
    if isinstance(sig, dict):
        s = sig.get("semantic") or sig.get("label") or sig.get("name")
    elif isinstance(sig, str):
        s = sig
    else:
        s = None
    if isinstance(s, str):
        s = s.strip().lower()
    return s or None

def _visual_bucket(obj: dict | None, shift: int = SINGLES_BUCKET_SHIFT) -> tuple | None:
    if not isinstance(obj, dict):
        return None
    src = obj.get("features", obj)

    keys_try_new = ["phash_primary","phash_secondary","phash_u","phash_v","phash_alpha","phash_edge"]
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
    keys_try_old = ["phash","phash_rgba"]
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

class GroupResultsWidget(QtWidgets.QWidget):
    request_pair_decision = QtCore.pyqtSignal(str, str)

    def __init__(self, main_window, use_external_info_panel=True, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.main = main_window
        self.project_root = None
        self.results = None
        self.features_dir = None
        self.groups = []
        self.group_pairs_map = {}
        self.member_item_map = {}
        self.group_id_map = {}
        self.sub_image_map_cache = LRUCache(capacity=50)
        self.mother_pixmap_cache = None
        self.feature_json_cache = None

        self.thread_pool = QtCore.QThreadPool()
        self._loading_uuid = None

        self.leftView  = BBoxGraphicsView(self)
        self.rightView = BBoxGraphicsView(self)

        self.leftView.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.rightView.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.leftView.auto_key_white = False
        self.rightView.auto_key_white = False

        self.leftView.setObjectName("sheetView")
        self.rightView.setObjectName("sheetView")
        self.leftView.setVisible(False)

        img_splitter = QtWidgets.QSplitter(self)
        img_splitter.addWidget(self.leftView)
        img_splitter.addWidget(self.rightView)
        img_splitter.setStretchFactor(0, 0)
        img_splitter.setStretchFactor(1, 1)
        img_splitter.setSizes([0, 1])
        self.img_splitter = img_splitter

        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("pairTree")
        self.tree.setHeaderLabels(["名稱", "UUID", "子圖ID", "bbox", "路徑"])
        self.tree.setVisible(False)
        self.tree.itemSelectionChanged.connect(self._on_pair_tree_select)
        self.tree.itemDoubleClicked.connect(self._on_pair_tree_select)
        self._on_tree_select = self._on_pair_tree_select

        self.infoPanel = None
        self._info_labels = None

        self.vsplit = QtWidgets.QSplitter(self)
        self.vsplit.setOrientation(QtCore.Qt.Vertical)
        self.vsplit.addWidget(img_splitter)
        self.vsplit.addWidget(self.tree)
        self.vsplit.setStretchFactor(0, 6)
        self.vsplit.setStretchFactor(1, 0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.vsplit)

    def attach_caches(self, feature_cache, pixmap_cache):
        """由 MainWindow 傳入共享的快取物件。"""
        self.feature_json_cache = feature_cache
        self.mother_pixmap_cache = pixmap_cache

    def attach_external_info_panel(self, panel: QtWidgets.QWidget, labels: tuple):
        """由 MainWindow 傳入右側的 infoPanel 與 7 個 QLabel。"""
        self.infoPanel = panel
        self._info_labels = labels

    def enable_compact_view(self, hide_tree: bool = True, hide_right_image: bool = False, show_info: bool = True):
        if hasattr(self, "tree"):
            self.tree.setVisible(not hide_tree)
        if hasattr(self, "rightView"):
            self.rightView.setVisible(not hide_right_image)
        if self.infoPanel is not None:
            self.infoPanel.setVisible(bool(show_info))
        def _apply():
            try:
                self.vsplit.setStretchFactor(0, 6)
                self.vsplit.setStretchFactor(1, 3 if not hide_tree else 0)
                if self.infoPanel is not None:
                    self.vsplit.setStretchFactor(2, 1)
                    self.vsplit.setSizes([6, (3 if not hide_tree else 0), 1])
                else:
                    self.vsplit.setSizes([6, (3 if not hide_tree else 0)])
            except Exception:
                pass
        QtCore.QTimer.singleShot(0, _apply)


    def _build_groups_from_pairs(self, results: dict) -> list[dict]:
        """
        用 pair 邊集合建圖，找 (uuid, sub_id) 的連通分量做群組。
        互斥：每個節點只會在一個群裡。群內同一張圖同一子圖只保留一次。
        同時回填每組的 pairs（只保留兩端都在該組的 pair）。
        """
        
        def _split_id(s):
            if "#sub_" in s:
                u, tail = s.split("#sub_", 1)
                try:
                    sid = int(tail)
                except ValueError:
                    sid = None
                return u, sid
            return s, None

        nodes = set()
        edges = []
        for p in results.get("pairs", []):
            la, lb = p.get("left_id"), p.get("right_id")
            if not la or not lb:
                continue
            a = _split_id(la)
            b = _split_id(lb)
            nodes.add(a); nodes.add(b)
            edges.append((a, b, p))

        def _aspect_of_member(u, sid):
            member_id = f"{u}#sub_{sid}" if sid is not None else u
            
            it = self.main.id2item.get(member_id)
            
            if not it:
                it = self.main.id2item.get(u)

            if not it or getattr(it, "rgba", None) is None:
                f = self._load_feat(u) or {}
                if sid is None:
                    bbox = f.get("bbox")
                    if not bbox or len(bbox) != 4:
                        dims = (f.get("dimensions") or {})
                        w, h = dims.get("width"), dims.get("height")
                    else:
                        _, _, w, h = bbox 
                    
                    if not w or not h: return None
                    return float(w) / max(1.0, float(h))

                else:
                    mother = f
                    bbox = None
                    for si in (mother.get("sub_images") or []):
                        si_sid = si.get("sub_id")
                        
                        if isinstance(si_sid, str):
                            if si_sid.startswith("sub_"):
                                try:
                                    si_sid = int(si_sid.split("_")[1])
                                except Exception:
                                    continue
                            else:
                                try:
                                    si_sid = int(si_sid)
                                except Exception:
                                    continue
                        if si_sid == sid: 
                            bbox = si.get("bbox"); break
                    if not bbox: return None
                    _, _, w, h = bbox
                
                if not w or not h: return None
                return float(w) / max(1.0, float(h))

            try:
                return crop_aspect_ratio(it.rgba, alpha_thr=SHAPE_ALPHA_THR)
            except Exception:
                return None

        def _shape_hash_alpha(u, sid):
            member_id = f"{u}#sub_{sid}" if sid is not None else u
            it = self.main.id2item.get(member_id) or self.main.id2item.get(u)
            if not it or getattr(it, "rgba", None) is None:
                return None
            try:
                return phash_from_canon_alpha(it.rgba, alpha_thr=SHAPE_ALPHA_THR, pad_ratio=CANON_PAD_SECONDARY)
            except Exception:
                return None

        def _compatible(a, b):
            (ua, sa), (ub, sb) = a, b
            ha = _shape_hash_alpha(ua, sa)
            hb = _shape_hash_alpha(ub, sb)
            shape_ok = (ha is not None and hb is not None and int((ha ^ hb).bit_count()) <= PHASH_SHAPE_MAX)

            ra = _aspect_of_member(ua, sa)
            rb = _aspect_of_member(ub, sb)
            ar_ok = (ra is not None and rb is not None and abs(ra - rb) <= ASPECT_TOL)

            return (shape_ok and ar_ok)

        if not nodes:
            return []

        parent = {}
        rank = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            rank.setdefault(ra, 0); rank.setdefault(rb, 0)
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for a, b, _ in edges:
            union(a, b)
        for n in nodes:
            find(n)

        comp_members = defaultdict(list)
        for (u, sid) in nodes:
            comp_members[find((u, sid))].append((u, sid))

        groups = []
        gid_seq = 1

        def _bbox_of(u, sid):
            if sid is None: 
                return None
            
            sub_map = self.sub_image_map_cache.get(u)
            
            if sub_map is None:
                mother = self._load_feat(u) or {} 
                if not mother.get("is_spritesheet", False):
                    self.sub_image_map_cache.put(u, {})
                    return None
                
                sub_map = {}
                for si in mother.get("sub_images") or []:
                    si_sid = si.get("sub_id")
                    if si_sid is None:
                        continue
                    
                    try:
                        sub_map[int(si_sid)] = si.get("bbox")
                    except (ValueError, TypeError):
                        pass
                
                self.sub_image_map_cache.put(u, sub_map)

            try:
                return sub_map.get(int(sid))
            except (ValueError, TypeError):
                return None

        for root, mems in comp_members.items():
            if len(mems) < 2:
                continue
            g_members = []
            seen_once = set()
            for (u, sid) in mems:
                key = (u, sid)
                if key in seen_once:
                    continue
                seen_once.add(key)
                g_members.append({"uuid": u, "sub_id": sid, "bbox": _bbox_of(u, sid)})

            g_pairs = []
            for a, b, p in edges:
                if find(a) == root and find(b) == root:
                    ua, sa = a; ub, sb = b
                    g_pairs.append({
                        "members": [
                            {"uuid": ua, "sub_id": sa, "bbox": _bbox_of(ua, sa)},
                            {"uuid": ub, "sub_id": sb, "bbox": _bbox_of(ub, sb)},
                        ],
                        "score": p.get("score", 1.0),
                    })

            groups.append({
                "group_id": f"comp_{gid_seq}",
                "members": g_members,
                "pairs": g_pairs,
            })
            gid_seq += 1

        return groups

    def select_member_by_uuid(self, uuid_: str, sub_id: int | str | None = None) -> bool:
        """
        在右側樹中尋找指定 uuid (+ sub_id) 的節點並選取。
        ★ 修正：即使樹狀結構中找不到該節點，也強制更新右側視圖，確保畫面有東西。
        """
        key = (uuid_, sub_id)
        leaf = self.member_item_map.get(key)
        
        meta = None
        if leaf:
            if hasattr(self, "tree") and self.tree:
                self.tree.setCurrentItem(leaf)
            meta = leaf.data(0, QtCore.Qt.UserRole)
        
        if not meta:
            feat = self._load_feat(uuid_) or {}
            p_uuid = feat.get("parent_uuid")
            
            target_sid = None
            if sub_id is not None:
                try: target_sid = int(sub_id)
                except: target_sid = str(sub_id)

            bbox = None
            if sub_id is not None:
                source_feat = self._load_feat(p_uuid) if p_uuid else feat
                for si in (source_feat.get("sub_images") or []):
                    si_sid = si.get("sub_id")
                    try:
                        if int(si_sid) == int(sub_id):
                            bbox = si.get("bbox")
                            break
                    except:
                        if str(si_sid) == str(sub_id):
                            bbox = si.get("bbox")
                            break

            meta = {
                "type": "member",
                "uuid": uuid_,
                "sub_id": sub_id,
                "parent_uuid": p_uuid,
                "bbox": bbox,
                "group_id": p_uuid
            }

        self._update_views_from_meta(meta)

        for btn_name in ("btn_open_location", "btn_open_folder"):
            btn = getattr(self, btn_name, None)
            if btn:
                btn.setEnabled(True)
                
        return True

    def set_project_root(self, root: str, cache_dir: str = None):
        self.project_root = root
        base = cache_dir if cache_dir else os.path.join(root, ".image_cache")
        self.features_dir = os.path.join(base, "features")
        self._custom_cache_dir = cache_dir
    
    def load_from_results(self):
        if not self.project_root:
            return
        
        base = getattr(self, "_custom_cache_dir", None)
        if not base:
            base = os.path.join(self.project_root, ".image_cache")
            
        p = os.path.join(base, "results.json")
        
        if not os.path.exists(p):
            self.tree.clear()
            self.leftView.clear()
            self.rightView.clear()
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                self.results = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load results.json: {e}")
            return

        self.groups = self.results.get("similarity_groups", []) or self._build_groups_from_pairs(self.results)

        self.group_id_map = {
            g.get("group_id"): g 
            for g in self.groups 
            if g.get("group_id")
        }

        self.group_pairs_map = {g.get("group_id"): g.get("pairs") or [] for g in self.groups}
        self._rebuild_tree()

    def _rebuild_tree(self):
        self.tree.clear()
        self.member_item_map.clear()
        self.sub_image_map_cache = LRUCache(capacity=50)

        if not self.groups:
            return
        for g in self.groups:
            gid = g.get("group_id")
            root = QtWidgets.QTreeWidgetItem(self.tree, [gid, "-", "-", "-", "-"])
            root.setData(0, QtCore.Qt.UserRole, {"type": "group", "group_id": gid})

            for m in g.get("members", []):
                uuid_ = m.get("uuid"); sid = m.get("sub_id"); bbox = m.get("bbox")
                leaf = QtWidgets.QTreeWidgetItem(root, [
                    ("子圖" if sid is not None else "散圖"),
                    (uuid_ or "")[:8],
                    (str(sid) if sid is not None else "-"),
                    (str(bbox) if bbox else "-"),
                    "-"
                ])
                leaf.setData(0, QtCore.Qt.UserRole, {"type":"member","group_id":gid,"uuid":uuid_,"sub_id":sid,"bbox":bbox})

                self.member_item_map[(uuid_, sid)] = leaf

            root.setExpanded(True)
        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
            self._on_pair_tree_select()


    def _resolve_parent_uuid_and_feature(self, members) -> Tuple[Optional[str], Optional[dict]]:
        for m in members:
            if m.get("sub_id") is not None:
                feat = self._load_feat(m.get("uuid"))
                if feat and feat.get("parent_uuid"):
                    parent_uuid = feat.get("parent_uuid")
                    mother = self._load_feat(parent_uuid)
                    return parent_uuid, mother
        return None, None
    
    def _load_feat(self, uuid_: str):
        cached_feat = self.feature_json_cache.get(uuid_)
        if cached_feat is not None:
            return cached_feat
        
        if hasattr(self.main, "_load_feature_json"):
            feat = self.main._load_feature_json(uuid_)
            if feat:
                self.feature_json_cache.put(uuid_, feat)
                return feat
        return None
        
    def features_iter(self):
        if not self.project_root:
            return
        features_dir = getattr(self, "features_dir", None)
        if not features_dir:
             base = getattr(self, "_custom_cache_dir", None) or os.path.join(self.project_root, ".image_cache")
             features_dir = os.path.join(base, "features")

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

            top.setData(0, QtCore.Qt.UserRole, {"type": "group", "parent_uuid": parent_uuid})

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

        for parent_uuid, data in groups.items():
            is_singles = (parent_uuid == SINGLES_GROUP_KEY)
            if is_singles:
                top = QtWidgets.QTreeWidgetItem(self.pair_tree, ["散圖", "-", "-", "-", "-"])
            else:
                mf = data["mother"] or {}
                title = mf.get("source_path") or f"sheet_{parent_uuid[:8]}"
                top = QtWidgets.QTreeWidgetItem(self.pair_tree, [f"Spritesheet: {title}",
                                                                f"{parent_uuid[:8]}", "-", "-", "-"])
            top.setData(0, QtCore.Qt.UserRole, {"type": "group", "parent_uuid": parent_uuid})

            for ph in data["pairs"]:
                A = self.id2item.get(ph.left_id); B = self.id2item.get(ph.right_id)
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

    def _mother_pixmap(self, parent_uuid: str) -> QtGui.QPixmap | None:
        """只從快取讀取，不執行硬碟 I/O，避免卡住 UI。"""
        if self.mother_pixmap_cache is None:
            return None
        return self.mother_pixmap_cache.get(parent_uuid)

    def _crop_from_sheet(self, parent_uuid: str, bbox) -> QtGui.QPixmap | None:
        pm = self._mother_pixmap(parent_uuid)
        if not pm or not bbox:
            return None
        x, y, w, h = [int(v) for v in bbox]
        r = QtCore.QRect(x, y, w, h).intersected(pm.rect())
        if r.isEmpty():
            return None
        return pm.copy(r)
    
    def _on_bg_image_loaded(self, uuid_, img):
        """背景圖片讀取完成後的回調"""
        if img.isNull():
            return

        pm = QtGui.QPixmap.fromImage(img)
        if self.mother_pixmap_cache:
            self.mother_pixmap_cache.put(uuid_, pm)

        if self._loading_uuid == uuid_:
            if hasattr(self, "_last_selected_meta"):
                self._update_views_from_meta(self._last_selected_meta)

    def _display_image_and_bboxes(self, pm, mother_feat, sub_id, meta):
        """輔助方法：顯示圖片與畫框"""
        self.rightView.show_image(pm, fit=True)
        
        target_bbox_coords = meta.get("bbox")
        if sub_id is not None:
            if not target_bbox_coords:
                for si in (mother_feat.get("sub_images") or []):
                    if str(si.get("sub_id")) == str(sub_id):
                        target_bbox_coords = si.get("bbox")
                        break
            
            if target_bbox_coords:
                self.rightView.draw_bboxes([{
                    "sub_id": str(sub_id), 
                    "bbox": target_bbox_coords
                }])
                self.rightView.focus_bbox(str(sub_id))

    def _update_views_from_meta(self, meta: dict | None):
        """根據 meta 字典更新右側視圖和資訊面板 (支援非同步)。"""
        self._last_selected_meta = meta
        self.rightView.clear()

        if not meta:
            self._update_info_panel(None, None, None)
            return

        if meta.get("type") == "group":
            gid = meta.get("group_id") or meta.get("parent_uuid")
            self._update_info_panel(None, None, gid)
            return
        
        uuid_  = meta.get("uuid")
        sub_id = meta.get("sub_id")
        gid    = meta.get("group_id")
        self.current_uuid = uuid_

        feat = self._load_feat(uuid_) or {}
        rel = feat.get("source_path")
        parent_uuid = feat.get("parent_uuid")
        
        target_load_uuid = parent_uuid if parent_uuid else uuid_
        self._loading_uuid = target_load_uuid 

        if not parent_uuid: 
            mother_feat = feat
        else:
            mother_feat = self._load_feat(parent_uuid) or {}
            if not rel: rel = mother_feat.get("source_path")

        pm = self._mother_pixmap(target_load_uuid)

        if pm and not pm.isNull():
            self._display_image_and_bboxes(pm, mother_feat, sub_id, meta)
        elif rel and self.project_root:
            abs_p = os.path.join(self.project_root, rel)
            if os.path.exists(abs_p):
                worker = LoaderRunnable(abs_p, target_load_uuid, WorkerSignals())
                worker.emitter.loaded.connect(self._on_bg_image_loaded)
                self.thread_pool.start(worker)
        
        self._update_info_panel(uuid_, sub_id if parent_uuid else None, gid)
        if hasattr(self, "btn_open_folder"):
            self.btn_open_folder.setEnabled(bool(rel))

    def _on_pair_tree_select(self):
        """群組樹／成員被點選時：右側顯示影像 + 更新資訊欄。"""
        it = self.tree.currentItem() if hasattr(self, "tree") else None
        if not it:
            self._update_views_from_meta(None)
            return

        meta = it.data(0, QtCore.Qt.UserRole) or {}
        self._update_views_from_meta(meta)

    def _show_one_side_sheet(self, view: BBoxGraphicsView, parent_uuid):
        view.clear()
        if not parent_uuid: return
        pm = self._mother_pixmap(parent_uuid)
        if not pm: return
        view.show_image(pm, fit=False)                   
        mf = self._load_feat(parent_uuid) or {}
        view.draw_bboxes(mf.get("sub_images") or [])

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


    def _member_rel_path(self, uuid_, sub_id) -> Optional[str]:
        feat = self._load_feat(uuid_)
        if not feat: return None
        return feat.get("source_path")
    
    def _show_member_views(self, uuid_: str, sub_id: int | None):
        """
        右側顯示：
        - 若為子圖：母圖 + 全部 bbox，並高亮目前 sub_id（同一張母圖時只切換 focus，不重畫）
        - 若為散圖：顯示自身影像
        """
        feat = self._load_feat(uuid_) or {}
        parent_uuid = feat.get("parent_uuid")

        if parent_uuid is not None:
            mother = self._load_feat(parent_uuid) or {}
            rel = mother.get("source_path")
            if rel and self.project_root:
                abs_path = os.path.join(self.project_root, rel)
                pm = QtGui.QPixmap(abs_path)
                if not pm.isNull():
                    self.rightView.show_image(pm, fit=True)
                    bboxes = []
                    for si in (mother.get("sub_images") or []):
                        sid = si.get("sub_id")
                        bb  = si.get("bbox")
                        if sid is None or not bb:
                            continue
                        bboxes.append({"sub_id": str(sid), "bbox": bb})
                    self.rightView.draw_bboxes(bboxes)
                    if sub_id is not None:
                        self.rightView.focus_bbox(str(sub_id), use_secondary=True)
            return

        self._current_parent_uuid = None
        self.rightView.clear()
        self.leftView.clear()

        rel = feat.get("source_path")
        if not (rel and self.project_root):
            return
        abs_path = os.path.join(self.project_root, rel)
        pm = QtGui.QPixmap(abs_path)
        if pm.isNull():
            return
        self.rightView.show_image(pm, fit=True)


    def _show_mother_with_all_bboxes(self, parent_uuid: str, highlight_sid: int | None = None):
        self.rightView.clear()
        if not parent_uuid:
            return
        pm = self._mother_pixmap(parent_uuid)
        if not pm or pm.isNull():
            QtWidgets.QMessageBox.warning(self, "提示", "無法載入母圖")
            return
        
        self.rightView.show_image(pm, fit=True)

        mother = self._load_feat(parent_uuid) or {}
        bboxes = []
        for si in (mother.get("sub_images") or []):
            sid = si.get("sub_id")
            bb = si.get("bbox")
            if sid is None or not bb:
                continue
            bboxes.append({"sub_id": str(sid), "bbox": bb})
        self.rightView.draw_bboxes(bboxes)
        if highlight_sid is not None:
            self.rightView.focus_bbox(str(highlight_sid), use_secondary=True)

    def _on_tree_select(self, *args):
        return self._on_pair_tree_select(*args)

    def _open_in_explorer(self):
        meta = getattr(self, "_last_selected_meta", None)
        if not meta:
            return

        uuid_ = meta.get("uuid") or meta.get("parent_uuid")
        if not uuid_:
            return

        feat = self._load_feat(uuid_) or {}
        rel_path = feat.get("source_path")

        if not rel_path:
            parent_uuid = feat.get("parent_uuid")
            if parent_uuid:
                mother = self._load_feat(parent_uuid) or {}
                rel_path = mother.get("source_path")

        abs_path = None
        if rel_path and self.project_root:
            abs_path = os.path.join(self.project_root, rel_path)

        if (not abs_path or not os.path.exists(abs_path)) and hasattr(self.main, "id2item"):
            item = self.main.id2item.get(uuid_)
            if item and getattr(item, "file_path", None):
                abs_path = item.file_path

        if not abs_path or not os.path.exists(abs_path):
            return

        try:
            if sys.platform.startswith("win"):
                subprocess.run(["explorer", "/select,", abs_path])
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", abs_path])
            else:
                folder = os.path.dirname(abs_path)
                subprocess.run(["xdg-open", folder])
        except Exception:
            pass

    def _build_info_panel(self) -> QtWidgets.QWidget:
        """右側資訊面板：使用 Grid Layout 並配合 SizePolicy 實現自動換行"""
        panel = QtWidgets.QWidget(self)
        
        panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        
        grid.setColumnStretch(1, 1)

        self.info_uuid   = QtWidgets.QLabel("-")
        # self.info_subid  = QtWidgets.QLabel("-")
        self.info_size   = QtWidgets.QLabel("-")
        # self.info_source = QtWidgets.QLabel("-")
        self.info_path   = QtWidgets.QLabel("-")
        self.info_count  = QtWidgets.QLabel("-")
        self.info_marked = QtWidgets.QLabel("-")
        
        target_labels = (
            self.info_uuid, self.info_size, 
            self.info_path, self.info_count,
            self.info_marked
        )

        for lbl in target_labels:
            lbl.setWordWrap(True)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

            lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
            lbl.setMinimumWidth(1)
            
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        def add_row(row, text, value_widget):
            lb = QtWidgets.QLabel(text)
            lb.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
            lb.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
            grid.addWidget(lb, row, 0)
            grid.addWidget(value_widget, row, 1)

        add_row(0, "UUID：",    self.info_uuid)
        # add_row(1, "子圖ID：",  self.info_subid)
        add_row(1, "尺寸：",    self.info_size)
        # add_row(3, "來源：",    self.info_source)
        add_row(2, "路徑：",    self.info_path)
        add_row(3, "同群數量：", self.info_count)
        add_row(4, "標記整圖：", self.info_marked)
        
        outer.addLayout(grid)
        outer.addStretch(1)
        
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_open_folder = QtWidgets.QPushButton("開啟檔案位置")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._on_open_location_clicked)
        
        sp = self.btn_open_folder.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Fixed)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self.btn_open_folder.setSizePolicy(sp)

        row.addWidget(self.btn_open_folder)
        outer.addLayout(row)

        return panel

    def _on_open_location_clicked(self):
        """以目前樹狀選取的成員，開啟來源檔案位置"""
        path_to_open = None
        uuid_ = None
        
        if hasattr(self, "tree") and self.tree.currentItem():
            meta = self.tree.currentItem().data(0, QtCore.Qt.UserRole) or {}
            uuid_ = meta.get("uuid")

        if uuid_ and self.project_root:
            feat = self._load_feat(uuid_) or {}
            rel = feat.get("source_path")
            if not rel:
                parent_uuid = feat.get("parent_uuid")
                if parent_uuid:
                    mother_feat = self._load_feat(parent_uuid) or {}
                    rel = mother_feat.get("source_path")
            
            if rel:
                path_to_open = os.path.join(self.project_root, rel)

        if not path_to_open and self.info_path.text() and self.info_path.text() != "-":
            text_path = self.info_path.text().replace("\u200b", "")
            
            if self.project_root and not os.path.isabs(text_path):
                path_to_open = os.path.join(self.project_root, text_path)
            else:
                path_to_open = text_path

        if not path_to_open:
            if hasattr(self.main, "list_files"):
                items = self.main.list_files.selectedItems()
                if items:
                    meta = items[0].data(QtCore.Qt.UserRole) or {}
                    rel = meta.get("rel")
                    if rel:
                        path_to_open = rel if os.path.isabs(rel) else os.path.join(self.project_root or "", rel)

        if not path_to_open or not os.path.exists(path_to_open):
            QtWidgets.QMessageBox.information(self, "提示", f"無法定位檔案路徑。\n{path_to_open}")
            return

        try:
            abs_path = os.path.normpath(path_to_open)
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", abs_path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", abs_path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(abs_path)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "開啟失敗", str(e))

    def _resolve_source_path(self, uuid_: str) -> str | None:
        """由 uuid 找到來源路徑（若為子圖則回退到母圖），回傳絕對路徑。"""
        if not getattr(self, "project_root", None):
            return None
            
        f_dir = getattr(self, "features_dir", None)
        if not f_dir:
            base = getattr(self, "_custom_cache_dir", None) or os.path.join(self.project_root, ".image_cache")
            f_dir = os.path.join(base, "features")
            
        p = os.path.join(f_dir, f"{uuid_}.json")


    def _open_current_in_explorer(self):
        """把目前右側顯示的成員（或你維護的 current 選擇）對應檔案位置打開。"""
        current_uuid = getattr(self, "current_uuid", None)
        if not current_uuid:
            return
        p = self._resolve_source_path(current_uuid)
        if not p:
            QtWidgets.QMessageBox.information(self, "找不到檔案", "此項目沒有來源檔案（可能是子圖）。")
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", p])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(p)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "開啟失敗", str(e))

    def _update_info_panel(self, uuid_: str | None, sub_id: int | None, group_id: str | None):
        """更新右側資訊面板內容 (加入路徑斷行處理)"""
        if not self._info_labels:
            return
        lab_uuid, lab_size, lab_path, lab_dups, lab_marked = self._info_labels
        
        def set_text_smart(widget, text):
            if not widget: return
            s = str(text) if text is not None else "-"
            s_display = s.replace("/", "/\u200b").replace("\\", "\\\u200b")
            widget.setText(s_display)

        if uuid_ is None and group_id:
            grp = next((g for g in (self.groups or []) if g.get("group_id") == group_id), None)
            mems = grp.get("members", []) if grp else []
            set_text_smart(lab_uuid,  "-")
            # set_text_smart(lab_child, "-")
            set_text_smart(lab_size,  "-")
            # set_text_smart(lab_origin, "群組")
            set_text_smart(lab_path,   group_id)
            set_text_smart(lab_dups,   len(mems) if mems else "-")
            return

        if not uuid_:
            for w in (lab_uuid, lab_size, lab_path, lab_dups):
                set_text_smart(w, "-")
            return

        feat = self._load_feat(uuid_) or {}
        set_text_smart(lab_uuid, uuid_)
        # set_text_smart(lab_child, sub_id if sub_id is not None else "-")

        if sub_id is None:
            dims = feat.get("dimensions") or {}
            set_text_smart(lab_size, f'{dims.get("width","-")}×{dims.get("height","-")}')
            rel = feat.get("source_path")
            # set_text_smart(lab_origin, "散圖")
            if rel:
                full_path = rel if os.path.isabs(rel) else os.path.join(self.project_root, rel)
                set_text_smart(lab_path, full_path)
            else:
                set_text_smart(lab_path, "-")
        else:
            # set_text_smart(lab_origin, "組圖")
            pu = uuid_  
            if pu:
                bbox = None
                if (hasattr(self, "_last_selected_meta") and 
                    self._last_selected_meta and
                    self._last_selected_meta.get("type") == "member" and
                    self._last_selected_meta.get("uuid") == uuid_ and
                    str(self._last_selected_meta.get("sub_id")) == str(sub_id)):
                    bbox = self._last_selected_meta.get("bbox")

                mother = self._load_feat(pu) or {}
                w, h = "-", "-"

                if bbox and len(bbox) == 4:
                    w, h = bbox[2], bbox[3]
                
                set_text_smart(lab_size, f"{w}×{h}")
                rel = mother.get("source_path")
                if rel:
                    full_path = rel if os.path.isabs(rel) else os.path.join(self.project_root, rel)
                    set_text_smart(lab_path, full_path)
                else:
                    set_text_smart(lab_path, "-")
            else:
                set_text_smart(lab_size, "-")
                set_text_smart(lab_path, "-")

        if group_id:
            grp = self.group_id_map.get(group_id)
            set_text_smart(lab_dups, len(grp.get("members", [])) if grp else "-")
        else:
            set_text_smart(lab_dups, "-")

        is_marked = False
        if hasattr(self.main, "index") and self.main.index:
            try:
                rel = self.main.index._uuid_to_rel.get(uuid_)
                if rel:
                    meta = self.main.index.data.get("image_map", {}).get(rel)
                    if meta and meta.get("force_whole"):
                        is_marked = True
            except: pass
        set_text_smart(lab_marked, "是" if is_marked else "否")
            
    def _members_of_group(self, group_id: str):
        for g in self.groups:
            if g.get("group_id") == group_id:
                return g.get("members") or []
        return []

    def _estimate_best_angle_within_group(self, uuid_, sub_id, members) -> str:
        current_id = f"{uuid_}#sub_{sub_id}" if sub_id is not None else uuid_
        me = self.main.id2item.get(current_id) or self.main.id2item.get(uuid_)

        if not me or getattr(me, "rgba", None) is None:
            return "-"
        hA = phash_from_canon_rgba(me.rgba, 1)
        best = 10**9; best_ang = None; best_other = None
        for m in members:
            member_id = f"{m.get('uuid')}#sub_{m.get('sub_id')}" if m.get('sub_id') is not None else m.get('uuid')
            if m.get("uuid") == uuid_ and (m.get("sub_id") == sub_id):
                continue

            other = self.main.id2item.get(member_id) or self.main.id2item.get(m.get('uuid'))
            if not other or getattr(other, "rgba", None) is None:
                continue
            d, ang = best_rot_hamming_fast(hA, other.rgba, alpha_thr=1, early_stop_at=None)
            if d < best:
                best, best_ang, best_other = d, ang, other
        if best_ang is None:
            return "-"
        return f"{best_ang:.1f}°（Δ={best}）"

    def _open_current_file(self):
        if not getattr(self, "_current_abs_path", None):
            return
        p = self._current_abs_path
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.run(["open", p], check=False)
            else:
                subprocess.run(["xdg-open", p], check=False)
        except Exception:
            pass

    def _reveal_current_file(self):
        meta = getattr(self, "_last_selected_meta", None) or {}
        uuid_ = meta.get("uuid")
        if not uuid_:
            return

        feat = self._load_feat(uuid_) or {}
        rel = None
        if meta.get("sub_id") is None:
            rel = feat.get("source_path")
        else:
            pu = feat.get("parent_uuid")
            if pu:
                m = self._load_feat(pu) or {}
                rel = m.get("source_path")

        if rel and self.project_root:
            abs_path = os.path.join(self.project_root, rel)
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(abs_path)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", abs_path])
                else:
                    subprocess.Popen(["xdg-open", os.path.dirname(abs_path)])
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "開啟失敗", str(e))

    def _fit_view(self):
        for v in (self.leftView, self.rightView):
            if v.scene():
                v.fitInView(v.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _on_decide_clicked(self):
        it = self.tree.currentItem()
        if not it: return
        meta = it.data(0, QtCore.Qt.UserRole) or {}
        if meta.get("type") != "member": return
        gid = meta["group_id"]; u = meta["uuid"]; sub = meta.get("sub_id")

        pair = None
        for p in self.group_pairs_map.get(gid, []):
            mm = p.get("members", [])
            for m in mm:
                if m.get("uuid") == u and m.get("sub_id") == sub:
                    pair = mm; break
            if pair: break
        if not pair or len(pair) != 2:
            for g in self.groups:
                if g.get("group_id") == gid:
                    others = [m for m in g.get("members", []) if not (m.get("uuid")==u and m.get("sub_id")==sub)]
                    if others:
                        a, b = ({"uuid":u,"sub_id":sub}, others[0])
                        self.request_pair_decision.emit(self._get_full_id(a), self._get_full_id(b))
                    return
            return
        a, b = pair[0], pair[1]
        self.request_pair_decision.emit(self._get_full_id(a), self._get_full_id(b))

    def _get_full_id(self, member_dict):
        """從 member dict 組合出完整的 id (e.g., uuid#sub_id)"""
        uuid_ = member_dict.get("uuid")
        sub_id = member_dict.get("sub_id")
        if sub_id is not None:
            return f"{uuid_}#sub_{sub_id}"
        return uuid_