import json
import pickle
from pathlib import Path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_object(path: Path, obj) -> None:
    ensure_parent(path)
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(path: Path):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def save_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
