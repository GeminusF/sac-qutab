from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import sac_qutab.cli as cli
from sac_qutab.cli import build_parser, cmd_run_all
from sac_qutab.runtime import atomic_write_json, write_jsonl


def test_every_cli_subcommand_has_working_help() -> None:
    parser = build_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert subparser_action.choices is not None
    for command in subparser_action.choices:
        with pytest.raises(SystemExit) as exit_info:
            parser.parse_args([command, "--help"])
        assert exit_info.value.code == 0


def test_training_cli_has_no_academy_allocation_arguments() -> None:
    parser = build_parser()
    help_text = parser._subparsers._group_actions[0].choices["train"].format_help()
    assert "administrative" not in help_text
    assert "authorized-training-window" not in help_text
    assert "weight-preflight" in help_text and "preflight" in help_text


def test_run_all_freezes_all_fifteen_runs_before_final_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    staged_detectors: set[str] = set()
    staged_ecdfs: set[str] = set()
    def fake_validate_preflight(cfg, *_args, **kwargs):
        assert kwargs["max_device_batch"] == 16
        return {"weight_preflights": {
            preset.name: {"checkpoint_path": f"{preset.name}.safetensors", "checkpoint_sha256": preset.checkpoint_sha256}
            for preset in cfg.experiment.model_presets
        }}
    monkeypatch.setattr(cli, "validate_hardware_preflight", fake_validate_preflight)
    monkeypatch.setattr(cli, "run_training", lambda *_args, **_kwargs: {"status": "complete"})

    def fake_evaluate(run, checkpoint, pairs, review, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        role = Path(pairs).stem
        if role == "final_test":
            assert events == ["freeze"]
            assert len(staged_detectors) == len(staged_ecdfs) == 15
        write_jsonl(output / "pair_metrics.jsonl", [{"pair_id": f"{run.model.name}-{run.seed}"}])
        atomic_write_json(output / "provenance.json", {
            "split_role": role, "model_name": run.model.name, "training_seed": run.seed,
            "checkpoint_sha256": "checkpoint", "resolved_run_sha256": "run", "review_freeze_sha256": "review",
            "control_ecdf_sha256": "ecdf", "protocol_sha256": "protocol" if role == "final_test" else None,
        })
        if role == "validation":
            atomic_write_json(output / "control_ecdf.json", {"role": "validation"})
        return {"pair_count": 1}

    def fake_fit(_rows, _cfg, provenance, output, _minimum):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema_version": "sac-detector-v3"})
        staged_detectors.add(f"{provenance['model_name']}-seed{provenance['training_seed']}")
        return {}

    def fake_freeze(_cfg, _splits, _pairs, _review, checkpoints, detectors, ecdfs, _preflight, output, *, resolved_runs):
        assert len(checkpoints) == len(detectors) == len(ecdfs) == 15
        assert set(checkpoints) == set(detectors) == set(ecdfs)
        assert set(resolved_runs) == set(checkpoints)
        assert all(run.batch_size_per_device == 16 for run in resolved_runs.values())
        staged_ecdfs.update(ecdfs)
        events.append("freeze")
        atomic_write_json(output, {"schema_version": "sac-experiment-protocol-v4"})
        return {}

    monkeypatch.setattr(cli, "evaluate_pairs", fake_evaluate)
    monkeypatch.setattr(cli, "fit_nested_detectors", fake_fit)
    monkeypatch.setattr(cli, "create_protocol_freeze", fake_freeze)
    monkeypatch.setattr(cli, "load_detector", lambda _path: {})
    monkeypatch.setattr(cli, "evaluate_nested_detectors", lambda *_args, **_kwargs: {"schema_version": "sac-detector-evaluation-v2"})
    monkeypatch.setattr(cli, "generate_primary_statistics", lambda _cfg, _dirs, output, **_kwargs: atomic_write_json(output, {"schema_version": "sac-primary-statistics-v3"}))
    monkeypatch.setattr(cli, "generate_report_artifacts", lambda *_args, **_kwargs: {})

    args = Namespace(
        config="configs/core.yaml",
        set=[f"project.output_dir={json.dumps(str(tmp_path / 'artifacts'))}"],
        preflight=str(tmp_path / "preflight.json"),
        pair_manifest=["validation=validation", "calibration=calibration", "final_test=final_test"],
        review_freeze=str(tmp_path / "review.json"),
        max_device_batch=16,
    )
    cmd_run_all(args)
    assert events == ["freeze"]
