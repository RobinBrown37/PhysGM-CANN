from pathlib import Path
import yaml
from easydict import EasyDict


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, overrides=()):
    def read(path, chain=()):
        path = Path(path).resolve()
        if path in chain:
            raise ValueError("Cyclic config inheritance")
        value = yaml.safe_load(path.read_text())
        parent = value.pop("_base_", None)
        return merge(read(path.parent/parent, (*chain, path)), value) if parent else value
    value = read(path)
    for assignment in overrides:
        key, text = assignment.split("=", 1)
        cursor = value
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(text)
    return EasyDict(value)


def plain_config(value):
    if isinstance(value, dict):
        return {k: plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(v) for v in value]
    return value
