"""Small, explicit persistence helpers for immutable protocols and raw records."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_bytes(value):
    return json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(canonical_bytes([value.shape, value.dtype.str]))
    digest.update(value.tobytes())
    return digest.hexdigest()


def fit_payload(fit):
    """Export a certificate without importing experiment selectors."""
    return {
        "support": fit.support,
        "scores": fit.scores,
        "source_bounds": fit.source_bounds,
        "prices": fit.prices,
        "upper": fit.upper,
        "lower": fit.lower,
        "status": fit.status,
        "runtime": fit.runtime,
        "diagnostics": fit.diagnostics,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(
        (
            json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
