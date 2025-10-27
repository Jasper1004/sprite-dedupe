import os, json, uuid
from PyQt5 import QtCore
from ..utils.atomic import atomic_write_json

class IndexStore:
    def __init__(self, project_root: str):
        self.root = project_root
        self.cache_dir = os.path.join(self.root, ".image_cache")
        self.path = os.path.join(self.cache_dir, "index.json")
        self.data = {"version":"1.0","last_scan_timestamp":None,"image_map":{}}
        self._uuid_to_rel = {}
        self.load()
        self._rebuild_uuid_index()

    def _rebuild_uuid_index(self) -> None:
        self._uuid_to_rel = {
            meta.get("uuid"): rel
            for rel, meta in self.data.get("image_map", {}).items()
            if meta.get("uuid")
        }

    def mark_clean_by_uuid(self, uuid_: str) -> bool:
        rel = self._uuid_to_rel.get(uuid_)
        if not rel:
            return False
        meta = self.data["image_map"].get(rel)
        if not meta:
            return False
        if meta.get("dirty_features"):
            meta["dirty_features"] = False
            return True
        meta.setdefault("dirty_features", False)
        return False

    def rel(self, abs_path: str) -> str:
        return os.path.relpath(abs_path, self.root)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self._rebuild_uuid_index()

    def save(self):
        self.data["last_scan_timestamp"] = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        atomic_write_json(self.path, self.data)

    def touch_file(self, abs_path: str) -> str:
        rel = self.rel(abs_path)
        st = os.stat(abs_path)
        m = self.data["image_map"].get(rel)
        if m:
            dirty = (m.get("last_modified") != int(st.st_mtime) or m.get("size") != int(st.st_size))
            m["last_modified"] = int(st.st_mtime)
            m["size"] = int(st.st_size)
            if dirty:
                m["dirty_features"] = True
            else:
                m.setdefault("dirty_features", False)
            m.setdefault("status", "active")
            if m.get("uuid"):
                self._uuid_to_rel[m["uuid"]] = rel
            return m["uuid"]
        else:
            uid = str(uuid.uuid4())
            self.data["image_map"][rel] = {
                "uuid": uid,
                "last_modified": int(st.st_mtime),
                "size": int(st.st_size),
                "status": "active",
                "dirty_features": True
            }
            self._uuid_to_rel[uid] = rel
            return uid