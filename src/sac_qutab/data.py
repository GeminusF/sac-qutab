from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .config import Config
from .runtime import atomic_write_json, read_jsonl, sha256_file, write_jsonl


@dataclass(frozen=True)
class Sample:
    sample_id: str
    relative_path: str
    label: int
    class_name: str
    width: int
    height: int
    mode: str
    bytes: int
    sha256: str
    dhash: str
    group_id: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Sample":
        return cls(**row)

    def row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sample_id(version: str, relative_path: str) -> str:
    return hashlib.sha256(f"{version}\0{relative_path}".encode()).hexdigest()


def image_dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(gray, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


class _BKNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}

    def insert(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indices.append(index)
            return
        if distance in self.children:
            self.children[distance].insert(value, index)
        else:
            self.children[distance] = _BKNode(value, index)

    def search(self, value: int, radius: int) -> Iterator[int]:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            yield from self.indices
        for edge, child in self.children.items():
            if distance - radius <= edge <= distance + radius:
                yield from child.search(value, radius)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _resolve_groups(rows: list[dict[str, Any]], threshold: int, group_map: dict[str, str] | None) -> dict[str, int]:
    uf = _UnionFind(len(rows))
    digest_first: dict[str, int] = {}
    tree: _BKNode | None = None
    near_edges = 0
    for i, row in enumerate(rows):
        if row["sha256"] in digest_first:
            uf.union(i, digest_first[row["sha256"]])
        else:
            digest_first[row["sha256"]] = i
        value = int(row["dhash"], 16)
        if tree is not None:
            for other in tree.search(value, threshold):
                uf.union(i, other)
                near_edges += 1
            tree.insert(value, i)
        else:
            tree = _BKNode(value, i)
    if group_map:
        named: dict[str, int] = {}
        by_id = {row["sample_id"]: i for i, row in enumerate(rows)}
        unknown = set(group_map) - set(by_id)
        if unknown:
            raise ValueError(f"group map references unknown sample IDs: {sorted(unknown)[:3]}")
        for sample_id, group in group_map.items():
            idx = by_id[sample_id]
            if group in named:
                uf.union(idx, named[group])
            else:
                named[group] = idx
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    for indices in components.values():
        gid = hashlib.sha256("\n".join(sorted(rows[i]["sha256"] for i in indices)).encode()).hexdigest()
        for i in indices:
            rows[i]["group_id"] = gid
    return {
        "exact_duplicate_files": len(rows) - len(digest_first),
        "near_duplicate_edges": near_edges,
        "group_count": len(components),
        "non_singleton_groups": sum(len(v) > 1 for v in components.values()),
    }


def build_manifest(cfg: Config, output: Path | None = None, *, allow_count_mismatch: bool = False) -> dict[str, Any]:
    root = cfg.data.root
    if not root.is_dir():
        raise FileNotFoundError(f"EuroSAT root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for label, class_name in enumerate(cfg.data.class_names):
        folder = root / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing class directory: {folder}")
        for path in sorted(folder.iterdir(), key=lambda p: p.name):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            relative = path.relative_to(root).as_posix()
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (cfg.data.native_size, cfg.data.native_size):
                    raise ValueError(f"expected RGB {cfg.data.native_size}x{cfg.data.native_size}: {relative}")
                dhash = f"{image_dhash(image):016x}"
                width, height = image.size
            rows.append({
                "sample_id": _sample_id(cfg.data.dataset_version, relative),
                "relative_path": relative,
                "label": label,
                "class_name": class_name,
                "width": width,
                "height": height,
                "mode": "RGB",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dhash": dhash,
                "group_id": "",
            })
    if not allow_count_mismatch and len(rows) != cfg.data.expected_samples:
        raise ValueError(f"expected {cfg.data.expected_samples} samples, found {len(rows)}")
    group_map = None
    if cfg.data.group_map_path is not None:
        raw = json.loads(cfg.data.group_map_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
            raise ValueError("group map must be a JSON object of sample_id -> group name")
        group_map = raw
    stats = _resolve_groups(rows, cfg.data.near_duplicate_hamming_threshold, group_map)
    rows.sort(key=lambda row: row["sample_id"])
    target = output or cfg.output_path(cfg.data.manifest_relpath)
    write_jsonl(target, rows)
    summary = {
        "schema_version": "sac-manifest-v1",
        "dataset_version": cfg.data.dataset_version,
        "source_url": cfg.data.source_url,
        "license": cfg.data.license,
        "count": len(rows),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in rows).items())),
        "grouping_version": cfg.data.grouping_version,
        "near_duplicate_hamming_threshold": cfg.data.near_duplicate_hamming_threshold,
        "group_map_sha256": sha256_file(cfg.data.group_map_path) if cfg.data.group_map_path else None,
        "manifest_sha256": sha256_file(target),
        **stats,
    }
    atomic_write_json(target.with_suffix(".summary.json"), summary)
    return summary


def load_manifest(path: str | Path) -> list[Sample]:
    samples = [Sample.from_row(row) for row in read_jsonl(path)]
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest sample IDs are not unique")
    return samples


def create_splits(cfg: Config, manifest_path: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or cfg.output_path(cfg.data.manifest_relpath)
    samples = load_manifest(manifest_path)
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_id].append(sample)
    roles = list(cfg.data.split_fractions)
    class_count = len(cfg.data.class_names)
    totals = np.zeros(class_count, dtype=int)
    group_counts: dict[str, np.ndarray] = {}
    for gid, members in by_group.items():
        counts = np.bincount([m.label for m in members], minlength=class_count)
        group_counts[gid] = counts
        totals += counts
    targets = {role: totals * cfg.data.split_fractions[role] for role in roles}
    assigned = {role: np.zeros(class_count, dtype=int) for role in roles}
    role_groups = {role: [] for role in roles}
    rng = random.Random(cfg.data.split_seed)
    order = list(by_group)
    rng.shuffle(order)
    order.sort(key=lambda gid: -len(by_group[gid]))
    for gid in order:
        counts = group_counts[gid]
        def cost(role: str) -> tuple[float, int]:
            before = np.square((assigned[role] - targets[role]) / np.maximum(targets[role], 1)).sum()
            after = np.square((assigned[role] + counts - targets[role]) / np.maximum(targets[role], 1)).sum()
            return float(after - before), roles.index(role)
        role = min(roles, key=cost)
        assigned[role] += counts
        role_groups[role].append(gid)
    sample_roles: dict[str, str] = {}
    for role in roles:
        for gid in role_groups[role]:
            for sample in by_group[gid]:
                sample_roles[sample.sample_id] = role
    if len(sample_roles) != len(samples):
        raise RuntimeError("not every sample received a split role")
    group_roles = {gid: role for role, gids in role_groups.items() for gid in gids}
    if len(group_roles) != len(by_group):
        raise RuntimeError("group leakage or missing group detected")
    artifact = {
        "schema_version": "sac-splits-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "grouping_version": cfg.data.grouping_version,
        "seed": cfg.data.split_seed,
        "fractions": cfg.data.split_fractions,
        "sample_roles": dict(sorted(sample_roles.items())),
        "counts": dict(sorted(Counter(sample_roles.values()).items())),
        "class_counts": {role: assigned[role].tolist() for role in roles},
        "group_counts": {role: len(role_groups[role]) for role in roles},
    }
    target = output or cfg.output_path(cfg.data.split_relpath)
    atomic_write_json(target, artifact)
    artifact["split_sha256"] = sha256_file(target)
    return artifact


def validate_split_contract(
    cfg: Config,
    manifest_path: str | Path,
    split_path: str | Path,
) -> tuple[list[Sample], dict[str, Any]]:
    """Load and fully authenticate the one frozen split contract."""
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    samples = load_manifest(manifest_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "manifest_sha256", "grouping_version", "seed", "fractions",
        "sample_roles", "counts", "class_counts", "group_counts",
    }
    if not isinstance(split, dict) or set(split) != expected_fields or split.get("schema_version") != "sac-splits-v1":
        raise ValueError("split schema fields or version differ from sac-splits-v1")
    if split.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("split manifest hash does not match the current data manifest")
    if split.get("grouping_version") != cfg.data.grouping_version:
        raise ValueError("split grouping version differs from configuration")
    if split.get("seed") != cfg.data.split_seed or split.get("fractions") != cfg.data.split_fractions:
        raise ValueError("split seed or fractions differ from configuration")

    expected_roles = set(cfg.data.split_fractions)
    roles = split.get("sample_roles")
    if not isinstance(roles, dict) or set(roles) != {sample.sample_id for sample in samples}:
        raise ValueError("split sample IDs differ from the manifest")
    if set(roles.values()) != expected_roles:
        raise ValueError("split roles differ from the configured train/validation/calibration/final-test roles")

    counts = dict(sorted(Counter(roles.values()).items()))
    if split.get("counts") != counts:
        raise ValueError("split role counts do not match sample assignments")
    class_counts = {role: [0] * len(cfg.data.class_names) for role in cfg.data.split_fractions}
    groups_by_role = {role: set() for role in cfg.data.split_fractions}
    by_group: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        role = roles[sample.sample_id]
        if not 0 <= sample.label < len(cfg.data.class_names) or sample.class_name != cfg.data.class_names[sample.label]:
            raise ValueError(f"manifest label contract is invalid for {sample.sample_id}")
        class_counts[role][sample.label] += 1
        groups_by_role[role].add(sample.group_id)
        by_group[sample.group_id].add(role)
    leaked = {group_id: values for group_id, values in by_group.items() if len(values) != 1}
    if leaked:
        raise ValueError(f"groups cross split roles: {list(leaked)[:3]}")
    if split.get("class_counts") != class_counts:
        raise ValueError("split class counts do not match sample assignments")
    group_counts = {role: len(groups_by_role[role]) for role in cfg.data.split_fractions}
    if split.get("group_counts") != group_counts:
        raise ValueError("split group counts do not match sample assignments")
    return samples, split


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load_native_image(root: Path, sample: Sample) -> torch.Tensor:
    path = root / sample.relative_path
    if sha256_file(path) != sample.sha256:
        raise ValueError(f"image digest changed: {sample.sample_id}")
    with Image.open(path) as image:
        return TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)


def preprocess_image(image: torch.Tensor, input_size: int = 224) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W]")
    resized = F.interpolate(image[None], size=(input_size, input_size), mode="bicubic", align_corners=False, antialias=True)[0]
    return (resized - IMAGENET_MEAN.to(resized)) / IMAGENET_STD.to(resized)


def _deterministic_flip(image: torch.Tensor, sample_id: str, seed: int, epoch: int, horizontal_p: float, vertical_p: float) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{sample_id}".encode()).digest()
    h = int.from_bytes(digest[:8], "big") / 2**64
    v = int.from_bytes(digest[8:16], "big") / 2**64
    if h < horizontal_p:
        image = image.flip(-1)
    if v < vertical_p:
        image = image.flip(-2)
    return image


class ManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, root: Path, samples: Sequence[Sample], input_size: int, *, train: bool, seed: int, horizontal_p: float, vertical_p: float) -> None:
        self.root = root
        self.samples = list(samples)
        self.input_size = input_size
        self.train = train
        self.seed = seed
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = load_native_image(self.root, sample)
        if self.train:
            image = _deterministic_flip(image, sample.sample_id, self.seed, self.epoch, self.horizontal_p, self.vertical_p)
        return preprocess_image(image, self.input_size), sample.label, sample.sample_id
