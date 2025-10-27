import os, json
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

class ActionsLogger:
    def __init__(self, project_root: str):
        self.path = os.path.join(project_root, ".image_cache", "actions.log")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, action: str, target: dict, details: dict | None = None, result: str = "success"):
        evt = {
            "ts": QtCore.QDateTime.currentDateTimeUtc().toString(Qt.ISODate),
            "actor": os.getenv("USERNAME") or os.getenv("USER") or "user",
            "action": action,
            "target": target,
            "details": details or {},
            "result": result
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")