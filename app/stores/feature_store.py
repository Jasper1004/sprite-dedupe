import os, json
from ..utils.atomic import atomic_write_json

class FeatureStore:
    def __init__(self, project_root: str):
        self.root = project_root
        self.dir = os.path.join(self.root, ".image_cache", "features")
        os.makedirs(self.dir, exist_ok=True)

    def path(self, uuid_: str) -> str:
        return os.path.join(self.dir, f"{uuid_}.json")

    def load(self, uuid_: str) -> dict | None:
        p = self.path(uuid_)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save(self, uuid_: str, payload: dict) -> None:
        atomic_write_json(self.path(uuid_), payload)