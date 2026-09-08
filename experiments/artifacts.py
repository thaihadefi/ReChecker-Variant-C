"""Experiment provenance and reloadable model bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.config import ExperimentConfig
from representation.variants import VARIANTS, VariantSpec


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(_json_value(value), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    os.replace(temporary, destination)


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_bytes(path: str | Path) -> int:
    target = Path(path)
    return sum(
        item.stat().st_size
        for item in target.parent.glob(f"{target.name}*")
        if item.is_file()
    )


def source_fingerprint(root: str | Path = ".") -> str:
    base = Path(root)
    paths = sorted((base / "rechecker").rglob("*.py")) + [base / "main.py"]
    source_dirs = ("config", "data", "models", "representation", "experiments")
    paths = [base / "main.py"]
    for folder in source_dirs:
        paths.extend(sorted((base / folder).rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_metadata() -> dict[str, object]:
    packages = {}
    for name in ("tensorflow", "keras", "gensim", "scikit-learn", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class FoldArtifactPaths:
    run_dir: Path
    fold: int
    variant: str

    @property
    def stem(self) -> str:
        return f"fold_{self.fold + 1:02d}_{self.variant}"

    @property
    def result(self) -> Path:
        return self.run_dir / "results" / f"{self.stem}.json"

    @property
    def weights(self) -> Path:
        return self.run_dir / "models" / f"{self.stem}.weights.h5"

    @property
    def embedding(self) -> Path:
        return self.run_dir / "embeddings" / f"{self.stem}.model"

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifests" / f"{self.stem}.json"

    def is_complete(self) -> bool:
        required = (self.result, self.weights, self.embedding, self.manifest)
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            return False
        try:
            result = read_json(self.result)
            manifest = read_json(self.manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            result.get("fold") == self.fold
            and result.get("variant") == self.variant
            and manifest.get("fold") == self.fold
            and manifest.get("variant") == self.variant
        )


def write_bundle_manifest(
    paths: FoldArtifactPaths,
    config: ExperimentConfig,
    spec: VariantSpec,
    threshold: float,
) -> None:
    def relative(path: Path) -> str:
        return str(path.relative_to(paths.run_dir))

    write_json(paths.manifest, {
        "schema_version": 1,
        "fold": paths.fold,
        "variant": spec.name,
        "embedding_kind": spec.embedding,
        "sequence_policy": spec.sequence,
        "uses_separator": spec.use_separator,
        "uses_segment_embedding": spec.use_segment_embedding,
        "segment_ids": {"W": 0, "C": 1, "SEP": 2, "PAD": 3},
        "threshold": threshold,
        "config": config.to_dict(),
        "weights": relative(paths.weights),
        "embedding": relative(paths.embedding),
    })


@dataclass
class LoadedModelBundle:
    model: object
    embedding: object
    config: ExperimentConfig
    spec: VariantSpec
    threshold: float


def load_model_bundle(manifest_path: str | Path) -> LoadedModelBundle:
    from models.models import build_model
    from representation.embeddings import load_embedding

    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    base = manifest_path.parent.parent
    config = ExperimentConfig.from_dict(manifest["config"])
    spec = VARIANTS[manifest["variant"]]
    embedding = load_embedding(base / manifest["embedding"], spec.embedding)
    model = build_model(config, spec)
    model.load_weights(base / manifest["weights"])
    return LoadedModelBundle(
        model=model,
        embedding=embedding,
        config=config,
        spec=spec,
        threshold=float(manifest["threshold"]),
    )
