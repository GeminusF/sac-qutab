from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config, config_digest
from .data import Sample, hamming_hex, load_native_image, validate_split_contract
from .runtime import atomic_write_bytes, atomic_write_json, canonical_json, read_jsonl, sha256_bytes, sha256_file, write_jsonl

FAMILIES = ("spectral_appearance", "texture", "resolution")
SEVERITIES = ("mild", "moderate", "severe")

def pair_condition_signature(cfg: Config) -> str:
    contract = {
        "transform_version": cfg.transforms.version,
        "spectral_sampling": cfg.transforms.spectral_sampling,
        "spectral_gain_radii": cfg.transforms.spectral_gain_radii,
        "texture_lambdas": cfg.transforms.texture_lambdas,
        "resolution_sizes": cfg.transforms.resolution_sizes,
        "native_size": cfg.data.native_size,
        "include_identity_sham": cfg.transforms.include_identity_sham,
    }
    return sha256_bytes(canonical_json(contract).encode())



def derived_seed(global_seed: int, sample_id: str, family: str, version: str) -> int:
    digest = hashlib.sha256(f"{global_seed}\0{sample_id}\0{family}\0{version}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _unit_direction(seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    return [float(v) for v in torch.empty(3).uniform_(-1, 1, generator=generator)]


def _choice(items: Sequence[Sample], key: str) -> Sample:
    if not items:
        raise ValueError(f"no eligible candidates for {key}")
    index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(items)
    return sorted(items, key=lambda s: s.sample_id)[index]


def _eligible_other(source: Sample, candidate: Sample, threshold: int) -> bool:
    return (
        candidate.sample_id != source.sample_id
        and candidate.group_id != source.group_id
        and candidate.sha256 != source.sha256
        and hamming_hex(candidate.dhash, source.dhash) > threshold
    )


def build_pair_manifest(cfg: Config, samples: Sequence[Sample], split: dict[str, Any], role: str, output: str | Path) -> dict[str, Any]:
    if role not in {"validation", "calibration", "final_test"}:
        raise ValueError("counterfactual role must be validation, calibration, or final_test")
    manifest_sha256 = sha256_file(cfg.output_path(cfg.data.manifest_relpath))
    split_sha256 = sha256_file(cfg.output_path(cfg.data.split_relpath))
    configuration_sha256 = config_digest(cfg)
    condition_signature = pair_condition_signature(cfg)
    roles: dict[str, str] = split["sample_roles"]
    targets = [s for s in samples if roles[s.sample_id] == role]
    training_by_class: dict[int, list[Sample]] = defaultdict(list)
    controls_by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if roles[sample.sample_id] == "train":
            training_by_class[sample.label].append(sample)
        if roles[sample.sample_id] == role:
            controls_by_class[sample.label].append(sample)
    rows: list[dict[str, Any]] = []
    for source in sorted(targets, key=lambda s: s.sample_id):
        donor_candidates = [s for s in training_by_class[source.label] if _eligible_other(source, s, cfg.data.near_duplicate_hamming_threshold)]
        donor = _choice(donor_candidates, f"{cfg.transforms.seed}:{source.sample_id}:texture-donor")
        control_candidates = [s for label, values in controls_by_class.items() if label != source.label for s in values if _eligible_other(source, s, cfg.data.near_duplicate_hamming_threshold)]
        control = _choice(control_candidates, f"{cfg.transforms.seed}:{source.sample_id}:negative-control")
        for family in FAMILIES:
            latent_seed = derived_seed(cfg.transforms.seed, source.sample_id, family, cfg.transforms.version)
            direction = _unit_direction(latent_seed) if family == "spectral_appearance" else None
            for severity in ("sham",) + SEVERITIES:
                if family == "spectral_appearance":
                    radius = 0.0 if severity == "sham" else cfg.transforms.spectral_gain_radii[severity]
                    params: dict[str, Any] = {"direction": direction, "radius": radius, "gains": [1.0 + radius * value for value in direction]}
                    donor_row = None
                elif family == "texture":
                    strength = 0.0 if severity == "sham" else cfg.transforms.texture_lambdas[severity]
                    params = {"lambda": strength}
                    donor_row = donor
                else:
                    size = cfg.data.native_size if severity == "sham" else cfg.transforms.resolution_sizes[severity]
                    params = {"internal_size": size, "restore_size": cfg.data.native_size, "interpolation": "bicubic_antialias"}
                    donor_row = None
                semantic = {
                    "source_id": source.sample_id,
                    "source_sha256": source.sha256,
                    "family": family,
                    "severity": severity,
                    "latent_seed": latent_seed,
                    "parameters": params,
                    "donor_id": donor_row.sample_id if donor_row else None,
                    "donor_sha256": donor_row.sha256 if donor_row else None,
                    "version": cfg.transforms.version,
                }
                cache_key = sha256_bytes(canonical_json(semantic).encode())
                pair_id = sha256_bytes(f"{role}\0{cache_key}".encode())
                rows.append({
                    "schema_version": "sac-pair-v2",
                    "config_sha256": configuration_sha256,
                    "manifest_sha256": manifest_sha256,
                    "split_sha256": split_sha256,
                    "condition_signature": condition_signature,
                    "pair_id": pair_id,
                    "role": role,
                    "source_id": source.sample_id,
                    "source_path": source.relative_path,
                    "source_sha256": source.sha256,
                    "source_group_id": source.group_id,
                    "label": source.label,
                    "class_name": source.class_name,
                    "family": family,
                    "severity": severity,
                    "latent_seed": latent_seed,
                    "parameters": params,
                    "donor_id": donor_row.sample_id if donor_row else None,
                    "donor_path": donor_row.relative_path if donor_row else None,
                    "donor_sha256": donor_row.sha256 if donor_row else None,
                    "donor_group_id": donor_row.group_id if donor_row else None,
                    "control_id": control.sample_id,
                    "control_path": control.relative_path,
                    "control_sha256": control.sha256,
                    "control_group_id": control.group_id,
                    "transform_version": cfg.transforms.version,
                    "cache_key": cache_key,
                    "cache_relpath": f"{cfg.transforms.cache_relpath.as_posix()}/{cache_key}.npy",
                    "cache_sha256": None,
                    "audit_condition": f"{family}:{severity}",
                })
    validate_severity_support(rows)
    target = Path(output)
    write_jsonl(target, rows)
    summary = {
        "schema_version": "sac-pair-summary-v2",
        "role": role,
        "row_count": len(rows),
        "source_count": len(targets),
        "pair_manifest_sha256": sha256_file(target),
        "config_sha256": configuration_sha256,
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "condition_signature": condition_signature,
        "transform_version": cfg.transforms.version,
        "label_conditioned_texture_donors": True,
    }
    atomic_write_json(target.with_suffix(".summary.json"), summary)
    return summary


def validate_severity_support(rows: Sequence[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_id"], row["family"])].append(row)
    required = {"sham", *SEVERITIES}
    for (source_id, family), values in grouped.items():
        by_severity = {row["severity"]: row for row in values}
        if set(by_severity) != required:
            raise ValueError(f"incomplete severity support for {source_id}/{family}")
        latent = {row["latent_seed"] for row in values}
        if len(latent) != 1:
            raise ValueError("latent intervention changes across severity")
        if family == "texture" and len({row["donor_id"] for row in values}) != 1:
            raise ValueError("texture donor changes across severity")
        if family == "spectral_appearance":
            distances = [sum((gain - 1.0) ** 2 for gain in by_severity[s]["parameters"]["gains"]) ** 0.5 for s in ("sham",) + SEVERITIES]
            if distances != sorted(distances):
                raise ValueError("spectral perturbation is not nested")


def apply_intervention(source: torch.Tensor, family: str, parameters: dict[str, Any], donor: torch.Tensor | None = None) -> torch.Tensor:
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("source must be RGB [3,H,W]")
    if family == "spectral_appearance":
        gains = torch.tensor(parameters["gains"], dtype=source.dtype, device=source.device)[:, None, None]
        return (source * gains).clamp(0, 1)
    if family == "texture":
        if donor is None or donor.shape != source.shape:
            raise ValueError("texture shift requires a same-shape donor")
        strength = float(parameters["lambda"])
        source_fft = torch.fft.fft2(source)
        donor_fft = torch.fft.fft2(donor)
        mixed_amplitude = (1 - strength) * source_fft.abs() + strength * donor_fft.abs()
        transformed = torch.fft.ifft2(mixed_amplitude * torch.exp(1j * torch.angle(source_fft))).real
        return transformed.clamp(0, 1)
    if family == "resolution":
        internal = int(parameters["internal_size"])
        restored = int(parameters["restore_size"])
        low = F.interpolate(source[None], size=(internal, internal), mode="bicubic", align_corners=False, antialias=True)
        return F.interpolate(low, size=(restored, restored), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
    raise ValueError(f"unknown intervention family: {family}")


def _npy_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, tensor.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    return buffer.getvalue()


def _expected_parameters(cfg: Config, family: str, severity: str, latent_seed: int) -> dict[str, Any]:
    if family == "spectral_appearance":
        radius = 0.0 if severity == "sham" else cfg.transforms.spectral_gain_radii[severity]
        direction = _unit_direction(latent_seed)
        return {"direction": direction, "radius": radius, "gains": [1.0 + radius * value for value in direction]}
    if family == "texture":
        return {"lambda": 0.0 if severity == "sham" else cfg.transforms.texture_lambdas[severity]}
    if family == "resolution":
        size = cfg.data.native_size if severity == "sham" else cfg.transforms.resolution_sizes[severity]
        return {"internal_size": size, "restore_size": cfg.data.native_size, "interpolation": "bicubic_antialias"}
    raise ValueError(f"unknown intervention family: {family}")


def validate_pair_manifest_contract(cfg: Config, pair_manifest: str | Path, *, require_cache: bool) -> dict[str, Any]:
    rows = read_jsonl(pair_manifest)
    if not rows:
        raise ValueError("pair manifest is empty")
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    split_path = cfg.output_path(cfg.data.split_relpath)
    samples, split = validate_split_contract(cfg, manifest_path, split_path)
    by_id = {sample.sample_id: sample for sample in samples}
    roles: dict[str, str] = split["sample_roles"]
    declared_roles = {row.get("role") for row in rows}
    if len(declared_roles) != 1 or next(iter(declared_roles)) not in {"validation", "calibration", "final_test"}:
        raise ValueError("pair manifest must contain exactly one non-training role")
    role = next(iter(declared_roles))
    expected_contract = {
        "schema_version": "sac-pair-v2",
        "config_sha256": config_digest(cfg),
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "condition_signature": pair_condition_signature(cfg),
        "role": role,
        "transform_version": cfg.transforms.version,
    }
    expected_sources = {sample.sample_id for sample in samples if roles[sample.sample_id] == role}
    if {row.get("source_id") for row in rows} != expected_sources:
        raise ValueError("pair manifest source population differs from the frozen split role")
    training_by_class: dict[int, list[Sample]] = defaultdict(list)
    role_by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if roles[sample.sample_id] == "train":
            training_by_class[sample.label].append(sample)
        if roles[sample.sample_id] == role:
            role_by_class[sample.label].append(sample)
    expected_auxiliaries: dict[str, tuple[Sample, Sample]] = {}
    for source_id in expected_sources:
        source = by_id[source_id]
        donors = [candidate for candidate in training_by_class[source.label] if _eligible_other(source, candidate, cfg.data.near_duplicate_hamming_threshold)]
        donor = _choice(donors, f"{cfg.transforms.seed}:{source.sample_id}:texture-donor")
        controls = [candidate for label, values in role_by_class.items() if label != source.label for candidate in values if _eligible_other(source, candidate, cfg.data.near_duplicate_hamming_threshold)]
        control = _choice(controls, f"{cfg.transforms.seed}:{source.sample_id}:negative-control")
        expected_auxiliaries[source_id] = donor, control
    pair_ids: set[str] = set()
    for row in rows:
        if any(row.get(key) != value for key, value in expected_contract.items()):
            raise ValueError(f"pair contract provenance mismatch: {row.get('pair_id')}")
        source = by_id.get(row.get("source_id"))
        if source is None or roles[source.sample_id] != role:
            raise ValueError("pair source is absent from the declared split role")
        family, severity = row.get("family"), row.get("severity")
        if family not in FAMILIES or severity not in {"sham", *SEVERITIES}:
            raise ValueError("pair family/severity is outside the frozen condition registry")
        latent_seed = derived_seed(cfg.transforms.seed, source.sample_id, family, cfg.transforms.version)
        parameters = _expected_parameters(cfg, family, severity, latent_seed)
        donor, control = expected_auxiliaries[source.sample_id]
        donor_row = donor if family == "texture" else None
        semantic = {
            "source_id": source.sample_id, "source_sha256": source.sha256, "family": family, "severity": severity,
            "latent_seed": latent_seed, "parameters": parameters, "donor_id": donor_row.sample_id if donor_row else None,
            "donor_sha256": donor_row.sha256 if donor_row else None, "version": cfg.transforms.version,
        }
        cache_key = sha256_bytes(canonical_json(semantic).encode())
        pair_id = sha256_bytes(f"{role}\0{cache_key}".encode())
        immutable = {
            "pair_id": pair_id, "source_path": source.relative_path, "source_sha256": source.sha256,
            "source_group_id": source.group_id, "label": source.label, "class_name": source.class_name,
            "latent_seed": latent_seed, "parameters": parameters,
            "donor_id": donor_row.sample_id if donor_row else None, "donor_path": donor_row.relative_path if donor_row else None,
            "donor_sha256": donor_row.sha256 if donor_row else None, "donor_group_id": donor_row.group_id if donor_row else None,
            "control_id": control.sample_id, "control_path": control.relative_path, "control_sha256": control.sha256,
            "control_group_id": control.group_id, "cache_key": cache_key,
            "cache_relpath": f"{cfg.transforms.cache_relpath.as_posix()}/{cache_key}.npy",
            "audit_condition": f"{family}:{severity}",
        }
        if any(row.get(key) != value for key, value in immutable.items()):
            raise ValueError(f"pair semantic/provenance mismatch: {row.get('pair_id')}")
        if pair_id in pair_ids:
            raise ValueError("pair manifest contains duplicate pair IDs")
        pair_ids.add(pair_id)
        if require_cache:
            load_cached_tensor(cfg, row)
    validate_severity_support(rows)
    return {"role": role, "rows": len(rows), "sources": len(expected_sources), "condition_signature": expected_contract["condition_signature"]}


def materialize_pair_cache(cfg: Config, pair_manifest: str | Path, output_manifest: str | Path | None = None) -> dict[str, Any]:
    validate_pair_manifest_contract(cfg, pair_manifest, require_cache=False)
    rows = read_jsonl(pair_manifest)
    by_id: dict[str, Sample] = {}
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    for row in read_jsonl(manifest_path):
        sample = Sample.from_row(row)
        by_id[sample.sample_id] = sample
    for row in rows:
        source_sample = by_id[row["source_id"]]
        source = load_native_image(cfg.data.root, source_sample)
        donor = load_native_image(cfg.data.root, by_id[row["donor_id"]]) if row["donor_id"] else None
        transformed = apply_intervention(source, row["family"], row["parameters"], donor)
        payload = _npy_bytes(transformed)
        cache_path = cfg.project.output_dir / row["cache_relpath"]
        atomic_write_bytes(cache_path, payload)
        row["cache_sha256"] = sha256_bytes(payload)
    target = Path(output_manifest or pair_manifest)
    write_jsonl(target, rows)
    contract = validate_pair_manifest_contract(cfg, target, require_cache=True)
    summary = {"schema_version": "sac-cache-summary-v2", "row_count": len(rows), "pair_manifest_sha256": sha256_file(target), "cache_valid": True, "condition_signature": contract["condition_signature"]}
    atomic_write_json(target.with_suffix(".summary.json"), summary)
    return summary


def load_cached_tensor(cfg: Config, row: dict[str, Any]) -> torch.Tensor:
    if not row.get("cache_sha256"):
        raise ValueError("pair cache has not been materialized")
    path = cfg.project.output_dir / row["cache_relpath"]
    payload = path.read_bytes()
    if sha256_bytes(payload) != row["cache_sha256"]:
        raise ValueError(f"cache digest mismatch: {row['pair_id']}")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (3, cfg.data.native_size, cfg.data.native_size):
        raise ValueError(f"invalid cached tensor shape/dtype: {row['pair_id']}")
    return torch.from_numpy(array.copy())


def validate_cache(cfg: Config, pair_manifest: str | Path) -> dict[str, Any]:
    contract = validate_pair_manifest_contract(cfg, pair_manifest, require_cache=True)
    return {"rows": contract["rows"], "sources": contract["sources"], "role": contract["role"], "condition_signature": contract["condition_signature"]}
