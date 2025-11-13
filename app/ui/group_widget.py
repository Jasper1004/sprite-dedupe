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
        self.main = main_window
        self.project_root = None
        self.results = None
        self.features_dir = None
        self.groups = []
        self.group_pairs_map = {}
        self.member_item_map = {}
        self.group_id_map = {}
        self.sub_image_map_cache = LRUCache(capacity=50)
        self.mother_pixmap_cache = LRUCache(capacity=20)
        self.feature_json_cache = LRUCache(capacity=100)

        self.leftView  = BBoxGraphicsView(self)
        self.rightView = BBoxGraphicsView(self)

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

        # buckets = defaultdict(list)

        # for uuid_, feat in self.features_iter():
        #     is_spritesheet = feat.get("is_spritesheet", False)
        #     parent_uuid = feat.get("parent_uuid")

        #     if parent_uuid is not None:
        #         continue
            
        #     if is_spritesheet:
        #         subs = feat.get("sub_images") or []
        #         for i, sub in enumerate(subs):
        #             sk = _get_sig(sub) or _visual_bucket(sub)
        #             if sk:
        #                 sid = sub.get("sub_id")
                        
        #                 if isinstance(sid, str):
        #                     if sid.startswith("sub_"):
        #                         try:
        #                             sid = int(sid.split("_")[1])
        #                         except Exception:
        #                             sid = i
        #                     else:
        #                         try:
        #                             sid = int(sid)
        #                         except Exception:
        #                             sid = i

        #                 if sid is None:
        #                     sid = i
                        
        #                 buckets[sk].append((uuid_, sid))
                        
        #     else:
        #         k = _get_sig(feat) or _visual_bucket(feat)
        #         if k:
        #             buckets[k].append((uuid_, None))

        # for members in buckets.values():
        #     for (u, sid) in members:
        #         nodes.add((u, sid))

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

        # for members in buckets.values():
        #     if len(members) < 2:
        #         continue
        #     for i in range(len(members)):
        #         for j in range(i + 1, len(members)):
        #             a = members[i]; b = members[j]
        #             if not _compatible(a, b):
        #                 continue
        #             edges.append((a, b, {"score": 1.0}))

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

        # def _bbox_of(u, sid):
        #     if sid is None: 
        #         return None
        #     mf = self._load_feat(u) or {}
        #     parent_uuid = mf.get("parent_uuid")
        #     if not parent_uuid:
        #         return None
        #     mother = self._load_feat(parent_uuid) or {}
        #     for si in mother.get("sub_images") or []:
        #         if si.get("sub_id") == sid:
        #             return si.get("bbox")
        #     return None

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
        """在右側樹中尋找指定 uuid (+ sub_id) 的 'member' 節點並選取；若找到回傳 True。"""
        if not hasattr(self, "tree") or self.tree is None:
            return False
        
        key = (uuid_, sub_id)
        leaf = self.member_item_map.get(key)
        
        if leaf:
            self.tree.setCurrentItem(leaf)

            meta = leaf.data(0, QtCore.Qt.UserRole) or {}
            self._update_views_from_meta(meta)
            # self._last_selected_meta = meta
            # self._on_pair_tree_select()

            for btn_name in ("btn_open_location", "btn_open_folder"):
                btn = getattr(self, btn_name, None)
                if btn:
                    btn.setEnabled(True)
            return True
        
        return False

        # for i in range(self.tree.topLevelItemCount()):
        #     root = self.tree.topLevelItem(i)
        #     if root is None:
        #         continue
        #     for j in range(root.childCount()):
        #         leaf = root.child(j)
        #         if leaf is None:
        #             continue
        #         meta = leaf.data(0, QtCore.Qt.UserRole) or {}
                
        #         if meta.get("type") != "member":
        #             continue
                    
        #         meta_uuid = meta.get("uuid")
        #         meta_sub_id = meta.get("sub_id")
                
        #         if meta_uuid != uuid_:
        #             continue
                    
        #         if meta_sub_id is None and sub_id is None:
        #             pass
        #         elif meta_sub_id is None or sub_id is None:
        #             continue
        #         else:
        #             try:
        #                 if int(meta_sub_id) != int(sub_id):
        #                     continue
        #             except (ValueError, TypeError):
        #                 if str(meta_sub_id) != str(sub_id):
        #                     continue
        #         self.tree.setCurrentItem(leaf)
        #         self._last_selected_meta = meta
        #         self._on_pair_tree_select()

        #         for btn_name in ("btn_open_location", "btn_open_folder"):
        #             btn = getattr(self, btn_name, None)
        #             if btn:
        #                 btn.setEnabled(True)
        #         return True
        # return False

    def set_project_root(self, root: str):
        self.project_root = root
        self.features_dir = os.path.join(root, ".image_cache", "features")

    def load_from_results(self):
        if not self.project_root:
            return
        p = os.path.join(self.project_root, ".image_cache", "results.json")
        if not os.path.exists(p):
            self.tree.clear(); self.leftView.clear(); self.rightView.clear(); return
        with open(p, "r", encoding="utf-8") as f:
            self.results = json.load(f)

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
        # self.mother_pixmap_cache.clear()
        # self.feature_json_cache.clear()

        self.mother_pixmap_cache = LRUCache(capacity=20)
        self.feature_json_cache = LRUCache(capacity=100)
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

    # def _load_feat(self, uuid_: str):
    #     if hasattr(self.main, "_load_feature_json"):
    #         return self.main._load_feature_json(uuid_)
    #     return None

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
        """
        逐一產生 (uuid, feature_dict)
        來源：<project_root>/.image_cache/features/*.json
        """
        if not self.project_root:
            return
        features_dir = os.path.join(self.project_root, ".image_cache", "features")
        if not os.path.isdir(features_dir):
            return
        for fn in os.listdir(features_dir):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(features_dir, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    feat = json.load(f)
            except Exception:
                continue
            uuid_ = feat.get("uuid") or fn[:-5]
            yield uuid_, feat

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
    
    # def _mother_pixmap(self, parent_uuid: str) -> QtGui.QPixmap | None:
    #     mf = self._load_feat(parent_uuid)
    #     if not mf or not mf.get("source_path") or not self.project_root:
    #         return None
    #     p = os.path.join(self.project_root, mf["source_path"])
    #     pm = QtGui.QPixmap(p)
    #     return pm if not pm.isNull() else None

    def _mother_pixmap(self, parent_uuid: str) -> QtGui.QPixmap | None:
        cached_pm = self.mother_pixmap_cache.get(parent_uuid)
        if cached_pm is not None:
            return cached_pm

        mf = self._load_feat(parent_uuid)
        if not mf or not mf.get("source_path") or not self.project_root:
            return None
        
        p = os.path.join(self.project_root, mf["source_path"])
        if not os.path.exists(p):
            return None
            
        pm = QtGui.QPixmap(p)

        if pm.isNull():
            return None
            
        MAX_SIZE = 2048
        if pm.width() > MAX_SIZE or pm.height() > MAX_SIZE:
            pm = pm.scaled(
                MAX_SIZE, MAX_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        self.mother_pixmap_cache.put(parent_uuid, pm)
        return pm

    def _crop_from_sheet(self, parent_uuid: str, bbox) -> QtGui.QPixmap | None:
        pm = self._mother_pixmap(parent_uuid)
        if not pm or not bbox:
            return None
        x, y, w, h = [int(v) for v in bbox]
        r = QtCore.QRect(x, y, w, h).intersected(pm.rect())
        if r.isEmpty():
            return None
        return pm.copy(r)

    def _update_views_from_meta(self, meta: dict | None):
        """(私有) 根據 meta 字典更新右側視圖和資訊面板。"""
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

        if sub_id is not None:
            # --- (這是原 _on_pair_tree_select 的子圖邏輯) ---
            parent_uuid = uuid_ 
            target_bbox_coords = meta.get("bbox")
            mother = self._load_feat(parent_uuid) or {}
            rel = mother.get("source_path")
            if rel and self.project_root:
                pm = self._mother_pixmap(parent_uuid)
                if pm:
                    self.rightView.show_image(pm, fit=True)
                    if target_bbox_coords:
                        original_sid_str = str(sub_id)
                        target_bbox_dict = {
                            "sub_id": original_sid_str, 
                            "bbox": target_bbox_coords
                        }
                        self.rightView.draw_bboxes([target_bbox_dict])
                        self.rightView.focus_bbox(original_sid_str)
            self._update_info_panel(uuid_, sub_id, gid)
            if hasattr(self, "btn_open_folder"):
                self.btn_open_folder.setEnabled(bool(rel))
            return
        
        # --- (這是原 _on_pair_tree_select 的散圖邏輯) ---
        feat = self._load_feat(uuid_) or {}
        rel = feat.get("source_path")
        if rel and self.project_root:
            abs_p = os.path.join(self.project_root, rel)
            
            # --- 【LRU 修正】---
            pm = self.mother_pixmap_cache.get(uuid_)
            if pm is None:
                if os.path.exists(abs_p):
                    pm = QtGui.QPixmap(abs_p)
                    if not pm.isNull():
                        # 錯誤：self.mother_pixmap_cache[uuid_] = pm
                        self.mother_pixmap_cache.put(uuid_, pm) # 正確：使用 .put()
            # --- 【LRU 修正結束】---

            if pm and not pm.isNull():
                self.rightView.show_image(pm, fit=True)
        
        self._update_info_panel(uuid_, None, gid)
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
    
    # def _on_pair_tree_select(self):
    #     """群組樹／成員被點選時：右側顯示影像 + 更新資訊欄。"""
    #     it = self.tree.currentItem() if hasattr(self, "tree") else None
    #     self._last_selected_meta = None
    #     self.rightView.clear()
    #     if not it:
    #         self._update_info_panel(None, None, None)
    #         return

    #     meta = it.data(0, QtCore.Qt.UserRole) or {}
    #     self._last_selected_meta = meta

    #     if meta.get("type") == "group":
    #         gid = meta.get("group_id") or meta.get("parent_uuid")
    #         self._update_info_panel(None, None, gid)
    #         return
        
    #     uuid_  = meta.get("uuid")
    #     sub_id = meta.get("sub_id")
    #     gid    = meta.get("group_id")
    #     self.current_uuid = uuid_

    #     if sub_id is not None:
    #         parent_uuid = uuid_ 
    #         target_bbox_coords = meta.get("bbox")

    #         mother = self._load_feat(parent_uuid) or {}
    #         rel = mother.get("source_path")

    #         if rel and self.project_root:
    #             pm = self._mother_pixmap(parent_uuid)
    #             if pm:
    #                 self.rightView.show_image(pm, fit=True)
                    
    #                 # 如果 meta 中有 bbox 資訊，就用它來繪圖
    #                 if target_bbox_coords:
    #                     original_sid_str = str(sub_id)
                        
    #                     # draw_bboxes 期望的格式是 dict list
    #                     target_bbox_dict = {
    #                         "sub_id": original_sid_str, 
    #                         "bbox": target_bbox_coords
    #                     }
                        
    #                     self.rightView.draw_bboxes([target_bbox_dict])
    #                     self.rightView.focus_bbox(original_sid_str)
                    
    #         self._update_info_panel(uuid_, sub_id, gid)
            
    #         # if rel and self.project_root:
    #         #     pm = self._mother_pixmap(parent_uuid)
    #         #     if pm:
    #         #         self.rightView.show_image(pm, fit=True)
                    
    #         #         all_bboxes = (mother.get("sub_images") or [])
    #         #         target_bbox = None
                    
    #         #         id_to_find = sub_id 

    #         #         for si in all_bboxes:
    #         #             if si.get("sub_id") == id_to_find:
    #         #                 target_bbox = si
    #         #                 break
                    
    #         #         if target_bbox:
    #         #             original_sid_str = str(id_to_find)
                        
    #         #             self.rightView.draw_bboxes([target_bbox]) #
    #         #             self.rightView.focus_bbox(original_sid_str) #
            
    #         # self._update_info_panel(uuid_, sub_id, gid)

    #         if hasattr(self, "btn_open_folder"):
    #             self.btn_open_folder.setEnabled(bool(rel))
    #         return

    #     # feat = self._load_feat(uuid_) or {} 
    #     # rel = feat.get("source_path")
    #     # if rel and self.project_root:
    #     #     abs_p = os.path.join(self.project_root, rel)
    #     #     pm = QtGui.QPixmap(abs_p)
    #     #     if not pm.isNull():
    #     #         self.rightView.show_image(pm, fit=True)
        
    #     # self._update_info_panel(uuid_, None, gid)

    #     feat = self._load_feat(uuid_) or {}
    #     rel = feat.get("source_path")
    #     if rel and self.project_root:
    #         abs_p = os.path.join(self.project_root, rel)
    #         pm = self.mother_pixmap_cache.get(uuid_)
    #         if pm is None:
    #             if os.path.exists(abs_p):
    #                 pm = QtGui.QPixmap(abs_p)
    #                 if not pm.isNull():
    #                     self.mother_pixmap_cache[uuid_] = pm
    #         if pm and not pm.isNull():
    #             self.rightView.show_image(pm, fit=True)
    #     self._update_info_panel(uuid_, None, gid)
    
    #     if hasattr(self, "btn_open_folder"):
    #         self.btn_open_folder.setEnabled(bool(rel))

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
        """右側資訊面板：不依賴 MainWindow 的 splitter_right，直接在本 widget 內建立。"""
        panel = QtWidgets.QWidget(self)

        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        form  = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.info_uuid   = QtWidgets.QLabel("-")
        self.info_subid  = QtWidgets.QLabel("-")
        self.info_size   = QtWidgets.QLabel("-")
        self.info_source = QtWidgets.QLabel("-")
        self.info_path   = QtWidgets.QLabel("-")
        self.info_count  = QtWidgets.QLabel("-")
        
        form.addRow("UUID：",       self.info_uuid)
        form.addRow("子圖ID：",     self.info_subid)
        form.addRow("尺寸：",       self.info_size)
        form.addRow("來源：",       self.info_source)
        form.addRow("路徑：",       self.info_path)
        form.addRow("同群數量：",    self.info_count)
        
        outer.addLayout(form)
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
        """以目前樹狀選取的成員，開啟來源檔案位置。"""
        it = self.tree.currentItem()
        if not it:
            return
        meta = it.data(0, QtCore.Qt.UserRole) or {}
        if meta.get("type") != "member":
            if it.childCount():
                it = it.child(0)
                meta = it.data(0, QtCore.Qt.UserRole) or {}
            else:
                return

        uuid_ = meta.get("uuid")
        if not uuid_ or not self.project_root:
            return

        feat = self._load_feat(uuid_) or {}
        rel = feat.get("source_path")
        if not rel:
            parent_uuid = feat.get("parent_uuid")
            if parent_uuid:
                mother_feat = self._load_feat(parent_uuid) or {}
                rel = mother_feat.get("source_path")

        if not rel:
            QtWidgets.QMessageBox.information(self, "找不到檔案", "此項目沒有來源檔案路徑。")
            return

        abs_path = os.path.join(self.project_root, rel)
        if not os.path.exists(abs_path):
            QtWidgets.QMessageBox.information(self, "檔案不存在", f"檔案路徑不存在：\n{abs_path}")
            return

        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(abs_path)])
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
        p = os.path.join(self.project_root, ".image_cache", "features", f"{uuid_}.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                feat = json.load(f)
        except Exception:
            return None

        rel = feat.get("source_path")

        if not rel:
            pu = feat.get("parent_uuid")
            if pu:
                mp = os.path.join(self.project_root, ".image_cache", "features", f"{pu}.json")
                if os.path.exists(mp):
                    try:
                        with open(mp, "r", encoding="utf-8") as mf:
                            mother = json.load(mf)
                        rel = mother.get("source_path")
                    except Exception:
                        rel = None

        if not rel:
            return None

        abspath = os.path.join(self.project_root, rel)
        return abspath if os.path.exists(abspath) else None


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
        """將目前選擇的成員或群組寫到右側 infoPanel。"""
        if not self._info_labels:
            return
        lab_uuid, lab_child, lab_size, lab_origin, lab_path, lab_dups = self._info_labels
        def set_(w, v): w.setText(str(v) if w else "-")

        if uuid_ is None and group_id:
            grp = next((g for g in (self.groups or []) if g.get("group_id") == group_id), None)
            mems = grp.get("members", []) if grp else []
            set_(lab_uuid,  "-")
            set_(lab_child, "-")
            set_(lab_size,  "-")
            set_(lab_origin, "群組")
            set_(lab_path,   group_id)
            set_(lab_dups,   len(mems) if mems else "-")
            return

        if not uuid_:
            for w in (lab_uuid, lab_child, lab_size, lab_origin, lab_path, lab_dups):
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
                # mother = self._load_feat(pu) or {}
                # w = h = "-"
                # for si in (mother.get("sub_images") or []):
                #     if si.get("sub_id") == sub_id:
                #         x, y, w, h = si.get("bbox", (0, 0, 0, 0))
                #         break
                # set_(lab_size, f"{w}×{h}")
                # rel = mother.get("source_path")

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
                # else:
                #     for si in (mother.get("sub_images") or []):
                #         if si.get("sub_id") == sub_id:
                #             x, y, w, h = si.get("bbox", (0, 0, 0, 0))
                #             break
                
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
            # grp = next((g for g in (self.groups or []) if g.get("group_id") == group_id), None)
            grp = self.group_id_map.get(group_id)
            set_(lab_dups, len(grp.get("members", [])) if grp else "-")
        else:
            set_(lab_dups, "-")



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