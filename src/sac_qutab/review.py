from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score

from .config import Config, config_digest
from .interventions import pair_condition_signature, validate_pair_manifest_contract
from .runtime import atomic_write_json, read_jsonl, sha256_file, write_jsonl
from .tracking import publish_review_freeze_to_wandb

DECISIONS = {"accept", "reject"}
_ATTESTATION_STATEMENT = "The two reviewers completed decisions independently, were blind to model outputs and condition identity, and were qualified to judge whether the intended EuroSAT class remained visually defensible."
_NO_HUMAN_REVIEW_LIMITATION = (
    "No human semantic-preservation review was performed. All predeclared transform conditions are retained by design; "
    "the synthetic review fixture is pipeline QA only and is not scientific evidence."
)


def _opaque_id(seed: int, pair_id: str, phase: str) -> str:
    return hashlib.sha256(f"{seed}\0{phase}\0{pair_id}".encode()).hexdigest()[:20]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_preview(npy_path: Path, png_path: Path) -> None:
    array = np.load(npy_path, allow_pickle=False)
    image = np.clip(np.moveaxis(array, 0, -1) * 255.0, 0, 255).astype(np.uint8)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(png_path)


def export_review(cfg: Config, pair_manifest: str | Path, output_dir: str | Path) -> dict[str, Any]:
    contract = validate_pair_manifest_contract(cfg, pair_manifest, require_cache=True)
    if contract["role"] != "validation":
        raise ValueError("review export requires the validation pair manifest")
    rows = [row for row in read_jsonl(pair_manifest) if row["severity"] != "sham"]
    by_source_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_source_class[row["class_name"]][row["source_id"]].append(row)
    chosen_sources: dict[str, list[str]] = {}
    for class_name, source_rows in by_source_class.items():
        ordered = sorted(source_rows, key=lambda sid: hashlib.sha256(f"{cfg.audit.sampling_seed}:{sid}".encode()).hexdigest())
        if len(ordered) < cfg.audit.samples_per_class_condition:
            raise ValueError(f"not enough validation sources for review class {class_name}")
        chosen_sources[class_name] = ordered[: cfg.audit.samples_per_class_condition]
    selected_ids = {sid for values in chosen_sources.values() for sid in values}
    final_rows = [row for row in rows if row["source_id"] in selected_ids]
    expected = len(cfg.data.class_names) * cfg.audit.samples_per_class_condition * 9
    if len(final_rows) != expected:
        raise ValueError(f"expected {expected} final review rows, found {len(final_rows)}")
    remaining = [row for row in rows if row["source_id"] not in selected_ids]
    remaining.sort(key=lambda row: hashlib.sha256(f"{cfg.audit.sampling_seed}:pilot:{row['pair_id']}".encode()).hexdigest())
    if len(remaining) < cfg.audit.pilot_rows:
        raise ValueError("not enough separate rows for the review pilot")
    pilot_rows = remaining[: cfg.audit.pilot_rows]
    out = Path(output_dir)
    key_rows: list[dict[str, Any]] = []
    for phase, phase_rows in (("pilot", pilot_rows), ("final", final_rows)):
        shared_rows: list[dict[str, Any]] = []
        ordered = sorted(phase_rows, key=lambda row: hashlib.sha256(f"{cfg.audit.sampling_seed}:order:{row['pair_id']}".encode()).hexdigest())
        for row in ordered:
            review_id = _opaque_id(cfg.audit.sampling_seed, row["pair_id"], phase)
            preview_rel = Path("images") / f"{review_id}.png"
            cache_path = cfg.project.output_dir / row["cache_relpath"]
            if sha256_file(cache_path) != row["cache_sha256"]:
                raise ValueError(f"missing or corrupt cache for {row['pair_id']}")
            _save_preview(cache_path, out / preview_rel)
            shared_rows.append({
                "review_id": review_id,
                "intended_class": row["class_name"],
                "transformed_image": preview_rel.as_posix(),
                "decision": "",
                "pilot_usable": "yes_or_no" if phase == "pilot" else "",
            })
            key_rows.append({
                "review_id": review_id, "phase": phase, "pair_id": row["pair_id"], "source_id": row["source_id"],
                "class_name": row["class_name"], "family": row["family"], "severity": row["severity"],
                "preview_relpath": preview_rel.as_posix(), "preview_sha256": sha256_file(out / preview_rel),
            })
        fields = ("review_id", "intended_class", "transformed_image", "decision", "pilot_usable")
        _write_csv(out / f"{phase}_reviewer_1.csv", shared_rows, fields)
        _write_csv(out / f"{phase}_reviewer_2.csv", shared_rows, fields)
    _write_csv(out / "adjudication.csv", [{"review_id": row["review_id"], "decision": ""} for row in key_rows if row["phase"] == "final"], ("review_id", "decision"))
    write_jsonl(out / "review_key.jsonl", key_rows)
    metadata = {
        "schema_version": "sac-review-export-v2",
        "config_sha256": config_digest(cfg),
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "review_key_sha256": sha256_file(out / "review_key.jsonl"),
        "condition_signature": pair_condition_signature(cfg),
        "sampling_seed": cfg.audit.sampling_seed,
        "samples_per_class_condition": cfg.audit.samples_per_class_condition,
        "reuse_sources_across_conditions": cfg.audit.reuse_sources_across_conditions,
        "pilot_rows": len(pilot_rows),
        "final_rows": len(final_rows),
        "reviewer_instructions": "Complete reviewer sheets separately. Reviewers see intended class and transformed image; family/severity/parameters/model outputs are hidden. Pilot must be completed and declared usable before final review.",
    }
    atomic_write_json(out / "export_metadata.json", metadata)
    return metadata


def _read_decisions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _by_review_id(rows: Sequence[dict[str, str]], expected_ids: set[str], label: str) -> dict[str, dict[str, str]]:
    result = {row.get("review_id", ""): row for row in rows}
    if set(result) != expected_ids or len(result) != len(rows):
        raise ValueError(f"{label} review IDs differ from the blinded export")
    return result

def _validate_sheet_display(rows: Mapping[str, dict[str, str]], keys: Mapping[str, dict[str, Any]], label: str) -> None:
    expected_fields = {"review_id", "intended_class", "transformed_image", "decision", "pilot_usable"}
    for review_id, row in rows.items():
        key = keys[review_id]
        if (set(row) != expected_fields or row.get("intended_class") != key["class_name"]
                or row.get("transformed_image") != key["preview_relpath"]):
            raise ValueError(f"{label} immutable display fields differ from the blinded export: {review_id}")


def _wilson(successes: int, total: int, confidence: float) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires observations")
    z = float(norm.ppf(0.5 + confidence / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _review_identities(attestation_path: str | Path) -> tuple[dict[str, Any], tuple[str, str], str]:
    attestation = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    if set(attestation) != {"schema_version", "statement", "reviewer_1", "reviewer_2", "adjudicator"} or attestation["schema_version"] != "sac-review-attestation-v1" or attestation["statement"] != _ATTESTATION_STATEMENT:
        raise ValueError("review attestation schema or independence statement is invalid")
    names: list[str] = []
    for role in ("reviewer_1", "reviewer_2", "adjudicator"):
        person = attestation[role]
        if not isinstance(person, dict) or set(person) != {"name", "qualified", "independent"}:
            raise ValueError(f"review attestation {role} is invalid")
        name = person["name"]
        if not isinstance(name, str) or not name.strip() or person["qualified"] is not True or person["independent"] is not True:
            raise ValueError(f"review attestation {role} requires a real qualified independent person")
        names.append(name.strip())
    if len(set(names)) != 3:
        raise ValueError("reviewers and adjudicator must be three distinct people")
    return attestation, (names[0], names[1]), names[2]


def _qualitative_panel(cfg: Config, directory: Path, resolved: Sequence[dict[str, Any]], pairs: dict[str, dict[str, Any]]) -> Path:
    selected: list[dict[str, Any]] = []
    for family in ("spectral_appearance", "texture", "resolution"):
        family_rows = [row for row in resolved if row["family"] == family]
        for decision in ("accept", "reject"):
            candidate = next((row for row in family_rows if row["final_decision"] == decision), None)
            if candidate is not None:
                selected.append(candidate)
    if not selected:
        raise ValueError("qualitative panel requires resolved review rows")
    tile_width, tile_height = 480, 284
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected[:6]):
        pair = pairs[row["pair_id"]]
        original_path = Path(cfg.data.root) / pair["source_path"]
        original = Image.open(original_path).convert("RGB").resize((208, 208), Image.Resampling.BICUBIC)
        transformed = Image.open(directory / row["preview_relpath"]).convert("RGB").resize((208, 208), Image.Resampling.BICUBIC)
        x = (index % 2) * tile_width + 16
        y = (index // 2) * tile_height + 8
        canvas.paste(original, (x, y + 20))
        canvas.paste(transformed, (x + 232, y + 20))
        draw.text((x + 72, y), "Original", fill="black")
        draw.text((x + 272, y), "Counterfactual", fill="black")
        draw.text((x, y + 234), f"{row['family']} / {row['severity']} / {row['final_decision']} / {row['class_name']}", fill="black")
    target = directory / "qualitative_audit_panel.png"
    canvas.save(target)
    return target


def _expected_review_phases(cfg: Config, pair_rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    candidates = [row for row in pair_rows if row["severity"] != "sham"]
    by_source_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_source_class[row["class_name"]][row["source_id"]].append(row)
    chosen: set[str] = set()
    for source_rows in by_source_class.values():
        ordered = sorted(source_rows, key=lambda sid: hashlib.sha256(f"{cfg.audit.sampling_seed}:{sid}".encode()).hexdigest())
        chosen.update(ordered[: cfg.audit.samples_per_class_condition])
    final_rows = [row for row in candidates if row["source_id"] in chosen]
    remaining = [row for row in candidates if row["source_id"] not in chosen]
    remaining.sort(key=lambda row: hashlib.sha256(f"{cfg.audit.sampling_seed}:pilot:{row['pair_id']}".encode()).hexdigest())
    return {**{row["pair_id"]: "final" for row in final_rows}, **{row["pair_id"]: "pilot" for row in remaining[: cfg.audit.pilot_rows]}}


def freeze_review(cfg: Config, review_dir: str | Path, pair_manifest: str | Path, output: str | Path, attestation_path: str | Path) -> dict[str, Any]:
    contract = validate_pair_manifest_contract(cfg, pair_manifest, require_cache=True)
    if contract["role"] != "validation":
        raise ValueError("semantic audit may be frozen only from validation pairs")
    directory = Path(review_dir)
    metadata = json.loads((directory / "export_metadata.json").read_text(encoding="utf-8"))
    key_path = directory / "review_key.jsonl"
    if (metadata.get("schema_version") != "sac-review-export-v2" or metadata["pair_manifest_sha256"] != sha256_file(pair_manifest)
            or metadata["config_sha256"] != config_digest(cfg) or metadata["condition_signature"] != pair_condition_signature(cfg)
            or metadata.get("review_key_sha256") != sha256_file(key_path)):
        raise ValueError("review export belongs to a different pair/config/condition/key contract")
    attestation, reviewer_names, adjudicator_name = _review_identities(attestation_path)
    pair_rows = read_jsonl(pair_manifest)
    pairs = {row["pair_id"]: row for row in pair_rows}
    expected_phases = _expected_review_phases(cfg, pair_rows)
    key_rows = read_jsonl(key_path)
    expected_key_fields = {"review_id", "phase", "pair_id", "source_id", "class_name", "family", "severity", "preview_relpath", "preview_sha256"}
    if any(set(row) != expected_key_fields or not isinstance(row["review_id"], str) or not isinstance(row["pair_id"], str) for row in key_rows):
        raise ValueError("review key rows have an invalid schema")
    keys: dict[str, dict[str, Any]] = {row["review_id"]: row for row in key_rows}
    if len(keys) != len(key_rows) or {row["pair_id"] for row in key_rows} != set(expected_phases):
        raise ValueError("review key rows differ from the deterministic blinded sample")
    for key in key_rows:
        pair_id = key["pair_id"]
        pair = pairs.get(pair_id)
        phase = expected_phases.get(pair_id)
        review_id = _opaque_id(cfg.audit.sampling_seed, pair_id, phase or "")
        preview_relpath = f"images/{review_id}.png"
        copied = {"source_id": pair["source_id"], "class_name": pair["class_name"], "family": pair["family"], "severity": pair["severity"]} if pair else {}
        if (phase != key["phase"] or review_id != key["review_id"]
                or preview_relpath != key["preview_relpath"] or any(key.get(field) != value for field, value in copied.items())
                or key["preview_sha256"] != sha256_file(directory / preview_relpath)):
            raise ValueError("review key mapping or preview differs from the deterministic blinded export")
    pilot_ids = {row["review_id"] for row in key_rows if row["phase"] == "pilot"}
    final_ids = {row["review_id"] for row in key_rows if row["phase"] == "final"}
    for reviewer in (1, 2):
        pilot = _by_review_id(_read_decisions(directory / f"pilot_reviewer_{reviewer}.csv"), pilot_ids, f"pilot reviewer {reviewer}")
        _validate_sheet_display(pilot, keys, f"pilot reviewer {reviewer}")
        if len(pilot) != cfg.audit.pilot_rows or any(row["pilot_usable"].strip().lower() != "yes" or row["decision"].strip().lower() not in DECISIONS for row in pilot.values()):
            raise ValueError("both reviewers must independently complete and accept every separate pilot row")
    final_1 = _by_review_id(_read_decisions(directory / "final_reviewer_1.csv"), final_ids, "final reviewer 1")
    final_2 = _by_review_id(_read_decisions(directory / "final_reviewer_2.csv"), final_ids, "final reviewer 2")
    _validate_sheet_display(final_1, keys, "final reviewer 1")
    _validate_sheet_display(final_2, keys, "final reviewer 2")
    adjudication = _by_review_id(_read_decisions(directory / "adjudication.csv"), final_ids, "adjudication")
    expected = len(cfg.data.class_names) * cfg.audit.samples_per_class_condition * 9
    if len(final_ids) != expected:
        raise ValueError(f"expected {expected} final review rows")
    r1: list[str] = []
    r2: list[str] = []
    resolved: list[dict[str, Any]] = []
    for review_id in sorted(final_ids):
        key = keys[review_id]
        a = final_1[review_id]["decision"].strip().lower()
        b = final_2[review_id]["decision"].strip().lower()
        if a not in DECISIONS or b not in DECISIONS:
            raise ValueError(f"both independent decisions required: {review_id}")
        r1.append(a)
        r2.append(b)
        if a == b:
            decision, adjudicated = a, False
        else:
            decision = adjudication[review_id]["decision"].strip().lower()
            if decision not in DECISIONS:
                raise ValueError(f"disagreement requires independent adjudication: {review_id}")
            adjudicated = True
        resolved.append({**key, "reviewer_1": a, "reviewer_2": b, "final_decision": decision, "adjudicated": adjudicated})
    raw_agreement = sum(a == b for a, b in zip(r1, r2, strict=True)) / len(r1)
    kappa = float(cohen_kappa_score(r1, r2, labels=["accept", "reject"]))
    if not np.isfinite(kappa) or kappa < cfg.audit.min_kappa:
        raise ValueError(f"review reliability gate failed: kappa={kappa}")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        condition = f"{row['family']}:{row['severity']}"
        by_condition[condition].append(row)
        by_condition_class[(condition, row["class_name"])].append(row)
    conditions: dict[str, Any] = {}
    included: list[str] = []
    for condition, values in sorted(by_condition.items()):
        accepted = sum(row["final_decision"] == "accept" for row in values)
        rate = accepted / len(values)
        class_rates: dict[str, Any] = {}
        class_veto = False
        for class_name in cfg.data.class_names:
            class_values = by_condition_class[(condition, class_name)]
            class_accept = sum(row["final_decision"] == "accept" for row in class_values)
            class_rate = class_accept / len(class_values)
            class_rates[class_name] = {"accepted": class_accept, "total": len(class_values), "rate": class_rate, "wilson": _wilson(class_accept, len(class_values), cfg.audit.confidence_level)}
            class_veto |= class_rate < cfg.audit.min_class_acceptance_rate
        eligible = rate >= cfg.audit.min_condition_acceptance_rate and not class_veto
        if eligible:
            included.append(condition)
        conditions[condition] = {"accepted": accepted, "total": len(values), "rate": rate, "wilson": _wilson(accepted, len(values), cfg.audit.confidence_level), "class_rates": class_rates, "class_veto": class_veto, "eligible": eligible}
    qualified_families = [family for family in ("spectral_appearance", "texture", "resolution") if f"{family}:mild" in included and f"{family}:severe" in included]
    if not qualified_families:
        raise ValueError("no family retains both mild and severe audit-qualified conditions; primary test is blocked")
    panel_path = _qualitative_panel(cfg, directory, resolved, pairs)
    artifact = {
        "schema_version": "sac-review-freeze-v2",
        "config_sha256": config_digest(cfg),
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "condition_signature": pair_condition_signature(cfg),
        "attestation_sha256": sha256_file(attestation_path),
        "attestation": attestation,
        "reviewer_names": list(reviewer_names),
        "adjudicator_name": adjudicator_name,
        "pilot_rows": len(pilot_ids),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "kappa_scope": cfg.audit.kappa_scope,
        "rules": {
            "exclusion_granularity": cfg.audit.exclusion_granularity,
            "min_kappa": cfg.audit.min_kappa,
            "min_condition_acceptance_rate": cfg.audit.min_condition_acceptance_rate,
            "min_class_acceptance_rate": cfg.audit.min_class_acceptance_rate,
            "disagreement_policy": cfg.audit.disagreement_policy,
            "downstream_description": "whole family×severity conditions only",
        },
        "conditions": conditions,
        "included_conditions": included,
        "qualified_primary_families": qualified_families,
        "resolved_rows": resolved,
        "qualitative_panel_path": str(panel_path),
        "qualitative_panel_sha256": sha256_file(panel_path),
    }
    atomic_write_json(output, artifact)
    artifact["freeze_sha256"] = sha256_file(output)
    publish_review_freeze_to_wandb(cfg, output, artifact)
    return artifact


def create_no_human_review_policy(cfg: Config, pair_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    """Freeze an explicit all-condition policy without representing synthetic QA as human review."""
    contract = validate_pair_manifest_contract(cfg, pair_manifest, require_cache=True)
    if contract["role"] != "validation":
        raise ValueError("the no-human-review policy must be bound to validation pairs")
    expected_conditions = {
        f"{family}:{severity}"
        for family in ("spectral_appearance", "texture", "resolution")
        for severity in ("mild", "moderate", "severe")
    }
    observed_conditions = {
        str(row["audit_condition"])
        for row in read_jsonl(pair_manifest)
        if row["severity"] != "sham"
    }
    if observed_conditions != expected_conditions:
        raise ValueError("validation pairs differ from the predeclared transform-condition grid")
    artifact = {
        "schema_version": "sac-condition-selection-policy-v1",
        "policy": "no_human_semantic_review",
        "human_semantic_review_performed": False,
        "synthetic_pipeline_qa_is_scientific_evidence": False,
        "selection_basis": "all_predeclared_transform_conditions",
        "config_sha256": config_digest(cfg),
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "condition_signature": pair_condition_signature(cfg),
        "included_conditions": sorted(expected_conditions),
        "excluded_conditions": [],
        "limitation": _NO_HUMAN_REVIEW_LIMITATION,
    }
    atomic_write_json(output, artifact)
    return artifact


def validate_review_freeze_contract(
    cfg: Config,
    freeze: Mapping[str, Any],
    *,
    validation_pair_manifest: str | Path | None = None,
) -> dict[str, Any]:
    if freeze.get("schema_version") == "sac-condition-selection-policy-v1":
        expected_conditions = {
            f"{family}:{severity}"
            for family in ("spectral_appearance", "texture", "resolution")
            for severity in ("mild", "moderate", "severe")
        }
        manifest_sha = freeze.get("pair_manifest_sha256")
        if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
                or any(character not in "0123456789abcdef" for character in manifest_sha)
                or freeze.get("policy") != "no_human_semantic_review"
                or freeze.get("human_semantic_review_performed") is not False
                or freeze.get("synthetic_pipeline_qa_is_scientific_evidence") is not False
                or freeze.get("selection_basis") != "all_predeclared_transform_conditions"
                or freeze.get("config_sha256") != config_digest(cfg)
                or freeze.get("condition_signature") != pair_condition_signature(cfg)
                or set(freeze.get("included_conditions", [])) != expected_conditions
                or freeze.get("excluded_conditions") != []
                or freeze.get("limitation") != _NO_HUMAN_REVIEW_LIMITATION):
            raise ValueError("no-human-review condition policy is invalid")
        if validation_pair_manifest is not None:
            contract = validate_pair_manifest_contract(cfg, validation_pair_manifest, require_cache=True)
            if (contract["role"] != "validation"
                    or freeze.get("pair_manifest_sha256") != sha256_file(validation_pair_manifest)):
                raise ValueError("condition policy belongs to a different validation pair manifest")
        return dict(freeze)
    if (freeze.get("schema_version") != "sac-review-freeze-v2"
            or freeze.get("config_sha256") != config_digest(cfg)
            or freeze.get("condition_signature") != pair_condition_signature(cfg)):
        raise ValueError("semantic review freeze belongs to a different config or condition contract")
    if validation_pair_manifest is not None:
        contract = validate_pair_manifest_contract(cfg, validation_pair_manifest, require_cache=True)
        if contract["role"] != "validation" or freeze.get("pair_manifest_sha256") != sha256_file(validation_pair_manifest):
            raise ValueError("semantic review freeze belongs to a different validation pair manifest")
    expected_rules = {
        "exclusion_granularity": cfg.audit.exclusion_granularity,
        "min_kappa": cfg.audit.min_kappa,
        "min_condition_acceptance_rate": cfg.audit.min_condition_acceptance_rate,
        "min_class_acceptance_rate": cfg.audit.min_class_acceptance_rate,
        "disagreement_policy": cfg.audit.disagreement_policy,
        "downstream_description": "whole family×severity conditions only",
    }
    if freeze.get("rules") != expected_rules or freeze.get("kappa_scope") != cfg.audit.kappa_scope:
        raise ValueError("semantic review rules differ from the frozen config")
    kappa = freeze.get("cohen_kappa")
    if not isinstance(kappa, (int, float)) or not np.isfinite(kappa) or float(kappa) < cfg.audit.min_kappa:
        raise ValueError("semantic review reliability no longer passes the frozen threshold")
    conditions = freeze.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("semantic review condition evidence is missing")
    expected_conditions = {f"{family}:{severity}" for family in ("spectral_appearance", "texture", "resolution") for severity in ("mild", "moderate", "severe")}
    if set(conditions) != expected_conditions:
        raise ValueError("semantic review condition set differs from the candidate contract")
    included: list[str] = []
    expected_total = len(cfg.data.class_names) * cfg.audit.samples_per_class_condition
    for condition, values in sorted(conditions.items()):
        accepted, total = values.get("accepted"), values.get("total")
        class_rates = values.get("class_rates")
        if (not isinstance(accepted, int) or total != expected_total or not isinstance(class_rates, dict)
                or set(class_rates) != set(cfg.data.class_names) or values.get("rate") != accepted / total):
            raise ValueError(f"semantic review aggregate is inconsistent: {condition}")
        class_veto = False
        class_accepted = 0
        for class_name in cfg.data.class_names:
            cell = class_rates[class_name]
            cell_accepted, cell_total = cell.get("accepted"), cell.get("total")
            if (not isinstance(cell_accepted, int) or cell_total != cfg.audit.samples_per_class_condition
                    or cell.get("rate") != cell_accepted / cell_total):
                raise ValueError(f"semantic review class aggregate is inconsistent: {condition}/{class_name}")
            class_accepted += cell_accepted
            class_veto |= cell["rate"] < cfg.audit.min_class_acceptance_rate
        eligible = accepted / total >= cfg.audit.min_condition_acceptance_rate and not class_veto
        if class_accepted != accepted or values.get("class_veto") is not class_veto or values.get("eligible") is not eligible:
            raise ValueError(f"semantic review eligibility is inconsistent: {condition}")
        if eligible:
            included.append(condition)
    qualified = [family for family in ("spectral_appearance", "texture", "resolution") if f"{family}:mild" in included and f"{family}:severe" in included]
    if freeze.get("included_conditions") != included or freeze.get("qualified_primary_families") != qualified or not qualified:
        raise ValueError("semantic review included conditions or qualified families were not derived from frozen evidence")
    return dict(freeze)


def apply_review_freeze(pair_rows: Sequence[dict[str, Any]], freeze: dict[str, Any]) -> list[dict[str, Any]]:
    if freeze.get("schema_version") not in {"sac-review-freeze-v2", "sac-condition-selection-policy-v1"}:
        raise ValueError("unsupported review or condition-selection policy")
    signatures = {row.get("condition_signature") for row in pair_rows}
    if signatures != {freeze.get("condition_signature")}:
        raise ValueError("reviewed condition signature differs from the consuming pair manifest")
    included = set(freeze["included_conditions"])
    return [row for row in pair_rows if row["severity"] == "sham" or row["audit_condition"] in included]
