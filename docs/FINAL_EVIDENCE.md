# SAC-QUTAB Final Pre-Training Evidence

> Historical evidence snapshot through 2026-09-04. Its academy A100/MIG assumptions and launch gates were superseded on 2026-09-07. It is not an active runbook or current readiness verdict; see [Training Readiness](training_readiness.md).

## 2026-09-09 completed-campaign addendum

**COMPLETED AND LOCALLY VERIFIED.** The active campaign trained or authenticated the full ResNet-50 and ViT-Small/16 × seeds 17, 29, and 43 matrix, froze protocol v5 before final-test evaluation, completed all six final evaluations and detector evaluations, generated primary statistics and publication artifacts, synchronized the recorded W&B runs, and produced a locally verified recovery archive.

| Reproducibility identifier | Frozen value |
|---|---|
| Effective base configuration SHA-256 | `cc73206d4ff490bebaa9440af5689decd4a93d08038df486dadd8dfe0deb449c` |
| Campaign SHA-256 | `d6dd3cabca656ef4414b165d82de1d9f45a228418a25e84e696eab9f248c7e8b` |
| No-human condition-policy SHA-256 | `18838e7446d37c172a3fc7cf686779bf7c4309612378c836261ad0deaae31d2a` |
| Authenticated final-test manifest SHA-256 | `10cb69bbcd74680af82e1ea10e2901e0c4a02270773b34b5c9f7648cd94b0308` |
| Frozen experiment protocol SHA-256 | `cd20ca70be0619e87b2422fa99f7601c9b86578e06475f15a385903c9ed134a9` |
| Final export archive SHA-256 | `8aa20619ea4d3865c7c44ec68f7007ba1d3225808853a7e0c27b96c684bc5ac1` |

Observed completion evidence is `FINALIZE_COMPLETED`, `FINAL_PIPELINE_FULLY_VERIFIED`, six runs, 68 generated report files, 47 PNG figures, 15 CSV tables, and `LOCAL_FINAL_BACKUP_FULLY_VERIFIED` for 424 internal files. The campaign, no-human policy, final-test manifest, and protocol hashes above were recalculated from the verified local extraction and matched exactly. The final local source was reconciled against the exported A100 source; the report-contract vocabulary fix and its regression fixture are byte-identical to the backup version. The complete local CPU-safe test suite then passed: **59 passed, 0 failed** (90 expected scikit-learn warnings from deliberately single-label confusion-matrix fixtures).

The exact recovery/verification commands, pinned A100 environment, W&B handling, freeze boundary, and canonical local archive paths are recorded in [A100_TRAINING_RUNBOOK.md](A100_TRAINING_RUNBOOK.md#8-completed-campaign-reproducibility-record). The historical pre-training evidence below remains unchanged as an audit trail and must not be read as the current project status.

## Verdict

**READY FOR FUTURE A100 PREFLIGHT**

This is a local codebase-readiness verdict, not hardware validation, authorization, or an empirical-result claim. The user reports that the future allocation is one 20 GB MIG instance on an A100 40 GB parent, but no A100 is currently accessible. The assigned-host administrative, EuroSAT, exact-weight, human-review, and measured-compute gates below must pass before the six training jobs begin. No training, dataset acquisition, model download, hardware preflight, checkpoint generation, final-test scoring, or expensive evaluation was run during this update.

## Environment

| Item | Observed value |
|---|---|
| Date / shell | 2026-09-03; Windows PowerShell |
| MIG update / shell | 2026-09-04; Windows PowerShell |
| OS | Windows 11 `10.0.26200`, AMD64 |
| Explicit test interpreter | `C:\Users\fereh\AppData\Local\Programs\Python\Python312\python.exe` |
| Python | 3.12.10, 64-bit MSC |
| Core packages | torch 2.13.0+cu126; torchvision 0.28.0+cu126; NumPy 1.26.4; Pillow 12.3.0; PyYAML 6.0.3; scikit-learn 1.9.0; SciPy 1.17.1 |
| Local accelerator | One NVIDIA GeForce RTX 4050 Laptop GPU; CUDA runtime 12.6 |
| A100 status | No A100 is currently accessible or used; the one-20 GB-MIG allocation is user-reported and all UUID/profile/memory/SM/performance facts remain unverified |
| Repository status | This directory has no `.git` metadata; `git status --short` exited 128 with “not a git repository,” so Git-based change/cleanliness evidence is unavailable |

The unqualified shell command `python` resolved to a separate Hermes Python 3.12 environment that had torch but no torchvision. All final executable evidence therefore uses the explicit interpreter above. Installing packages was neither necessary nor performed.

## Commands and actual outcomes

PowerShell set `PYTHONPATH=src` for Python commands. `-p no:cacheprovider` avoided cache writes; pytest itself was run outside the workspace sandbox because the sandbox denied creation of its Windows temporary directories.

| Order | Command or command group | Exit / observed result |
|---|---|---|
| Baseline | `python -m compileall -q src tests` and `python -m pytest --collect-only -q` | Exit 0; 31 pre-correction tests collected. |
| Baseline targeted A | `python -m pytest -q -p no:cacheprovider tests/test_config_data.py tests/test_governance.py tests/test_interventions_review.py tests/test_cli.py --maxfail=20` | Exit 0 outside sandbox; 18 passed. Earlier sandbox attempts failed at pytest temporary-directory setup and did not establish test failures. |
| Baseline targeted B | `python -m pytest -q -p no:cacheprovider tests/test_models_train.py tests/test_metrics_statistics.py tests/test_evaluation_reporting.py --maxfail=20` | Exit 0; 13 passed, 36 warnings. |
| Wrong-interpreter check | `python -m pytest ...` after edits | Exit 1 during collection; five `ModuleNotFoundError: torchvision` errors in the unrelated shell environment. No tests executed; resolved by selecting the pinned-capable Python 3.12 interpreter. |
| Post-edit focused set | Explicit Python 3.12 `-m pytest` over six modified-component files | Exit 1; 34 passed, two new tests failed because their fixtures retained a returned convenience digest and omitted a `pytest` import. This was not a product defect. |
| Fixture-only affected rerun | Explicit Python 3.12 `-m pytest -q -p no:cacheprovider tests/test_config_data.py tests/test_models_train.py` | Exit 0; 10 passed. |
| Static check | Explicit Python 3.12 `-m compileall -q src tests` | Exit 0; no output. No repository lint/type command is configured in `pyproject.toml`, so none was invented. |
| Config matrices | Explicit Python 3.12 `-m sac_qutab validate-config --config configs/core.yaml --world-size 1` and the same with `--world-size 2` | Both exit 0 after the final config correction; each emits six runs at global batch 64 (one rank 32×2, two ranks 32×1). |
| Complete CPU-safe suite | Explicit Python 3.12 `-m pytest -q -p no:cacheprovider` | Exit 0; 39 passed, 0 failed, 0 skipped, 0 xfailed; 36 warnings; 72.15 s. |
| Timing-default affected tests | Explicit Python 3.12 `-m pytest -q -p no:cacheprovider tests/test_config_data.py tests/test_governance.py tests/test_models_train.py` | Exit 0; 19 passed in 12.87 s. |
| Allowed final full rerun | Explicit Python 3.12 `-m pytest -q -p no:cacheprovider` | Exit 0; 39 passed, 0 failed, 0 skipped, 0 xfailed; 36 warnings; 66.20 s. This was the sole additional full run, required after correcting the frozen dry-run bounds from 128/2 to 1,600/25. |
| Final bounded scan | `rg` scan of `src`, `tests`, `configs`, README and active design/runbook/traceability/validation docs | Exit 0; 0 incomplete markers and 0 stale-contract markers. The sole bare `pass` is the intentional body of `MetricUndefinedError`. |
| MIG-update initial focused attempt | `python -m pytest tests/test_governance.py tests/test_models_train.py -q` | Exit 1 during collection; two `ModuleNotFoundError: sac_qutab` errors because this checkout is not installed and `PYTHONPATH` was not set. No test executed. |
| MIG-update sandbox attempts | Focused pytest with `PYTHONPATH=src`, cache disabled, and fresh system/workspace base-temp paths | Exit 1 before useful results; the filesystem sandbox denied pytest access to both temporary roots. These runs did not establish product failures. |
| MIG-update focused set | `python -m pytest tests/test_governance.py tests/test_models_train.py -q -p no:cacheprovider` with `PYTHONPATH=src`, fresh workspace base-temp, outside the filesystem sandbox | Exit 0; 18 passed in 8.42 s. |
| MIG-update expanded affected set | Same invocation over `test_governance.py`, `test_models_train.py`, and `test_cli.py` | Exit 0; 25 passed in 18.30 s. |
| Final directly affected checks | Selector-count and transfer-policy node IDs | Exit 0; 4 passed in 4.02 s. |
| MIG-update static/config checks | `python -m compileall -q src tests`; `python -m sac_qutab validate-config --config configs/core.yaml --world-size 1` | Both exit 0; the config command emitted the exact six-run matrix at device batch 32, accumulation 2, effective global batch 64. No lint/type command is configured. |
| MIG-update complete CPU-safe suite | `python -m pytest -q -p no:cacheprovider` with `PYTHONPATH=src`, fresh workspace base-temp, outside the filesystem sandbox | Exit 0; **47 passed, 0 failed, 0 skipped, 0 xfailed; 36 warnings; 58.20 s.** This was the complete suite's sole run for the MIG update. |
| MIG-update final bounded scan | `rg` contract/incomplete-marker scan across `src`, `tests`, `configs`, README and `docs`, plus authoritative-file timestamp inventory | Exit 0; no stale canonical GPU field, obsolete multi-device launch, old artifact name, or unresolved core marker. Legacy v1 strings occur only in migration guards/tests. Authoritative source files retain their prior timestamps. |

The 36 warnings in both recorded full-suite runs all come from `tests/test_evaluation_reporting.py`: scikit-learn warns that its intentionally one-class fixture has a single label in `y_true` and `y_pred`. The reporting code passes the full label registry to `confusion_matrix`; no test failed and no warning was suppressed.

## Changed-file summary

| Area | Files | Purpose |
|---|---|---|
| Runtime/scientific contracts | `configs/core.yaml`; `src/sac_qutab/governance.py`; `train.py`; `statistics.py`; `reporting.py`; `cli.py` | Existing scientific corrections plus MIG-aware v2 governance, opaque selectors, one-rank/18 GiB admission, device provenance, production-loader dry runs, non-blocking CUDA transfer, reusable validation loading, and validation-inclusive projection. |
| Acceptance tests | `tests/test_config_data.py`; `test_governance.py`; `test_interventions_review.py`; `test_models_train.py`; `test_metrics_statistics.py`; `test_evaluation_reporting.py` | Existing scientific contracts plus v1 migration refusal, MIG/numeric selectors, topology/memory/device provenance, validation arithmetic, loader policy and 32×2→16×4→8×8 batch invariance. |
| Team documentation | `README.md`; `docs/A100_TRAINING_RUNBOOK.md`; `DESIGN_AND_TRAINING_RATIONALE.md`; `IMPLEMENTATION_PLAN.md`; `REQUIREMENTS_TRACEABILITY.md`; `VALIDATION_CLOSURE.md` | Reconciled v2 schemas, `compute-mig1.json`, one-MIG commands, loader/fallback policy, validation-inclusive timing, deferred worker comparison, and hardware-unverified claim boundary. |
| New closure records | `docs/SCIENTIFIC_CORRECTIONS_CLOSURE.md`; `REVIEW_CLOSURE.md`; this file | Frozen-finding ledger, one bounded verification pass, exact local evidence and handoff verdict. |

The five authoritative project/reference files and both PDFs were read but not modified.

## Findings and deferrals

Resolved frozen findings: **B-01, B-02, B-03, B-04, B-05, H-06, H-07, H-08**. Exact severities, criteria, supporting files, tests, and dispositions are in `SCIENTIFIC_CORRECTIONS_CLOSURE.md`. No blocker or high-severity item remains unresolved.

Deferred, without fabricated evidence:

- real approval, roster, active window, operator/storage record, reconciled allocation, and observed MIG selector/profile;
- EuroSAT acquisition, 27,000-image manifest, duplicate/group structure, and any geographic metadata;
- authorized ResNet18 and DeiT-Tiny files and full validated hashes;
- two qualified reviewers, independent adjudicator, signed attestation, decisions, κ and accepted conditions;
- one-MIG identity/memory/SM count, timing/recovery evidence, worker comparison, and interruption drill;
- six trained checkpoints, validation/calibration artifacts, global protocol freeze, one-time final test, metrics and multi-seed conclusions;
- final paper, slides, contribution report, repository URL/tag and submission.

Each item blocks the corresponding real execution or publication stage but does **not** block pre-training codebase readiness.

## Exact future one-MIG step

Nothing in this section was executed locally. First satisfy the non-hardware gates in `A100_TRAINING_RUNBOOK.md`: a real active `sac-administrative-v2` JSON, authenticated EuroSAT manifest/split and pair caches, completed human review freeze, and both `validate-weights` records with their exact checkpoint files still present. During the approved window, observe the real UUID/profile with `nvidia-smi -L`, populate the private record, and run the single-process bounded jobs—still not full training:

```bash
export CUDA_VISIBLE_DEVICES="$SAC_MIG_UUID"

python -m sac_qutab train --config configs/core.yaml \
  --model resnet18 --seed 17 \
  --run-dir artifacts/preflight/resnet18-seed17-mig1 \
  --administrative "$ADMINISTRATIVE_JSON" \
  --weight-preflight artifacts/preflight/resnet18_weights.json \
  --dry-run --authorized-training-window

python -m sac_qutab train --config configs/core.yaml \
  --model deit_tiny_patch16_224 --seed 17 \
  --run-dir artifacts/preflight/deit_tiny_patch16_224-seed17-mig1 \
  --administrative "$ADMINISTRATIVE_JSON" \
  --weight-preflight artifacts/preflight/deit_tiny_weights.json \
  --dry-run --authorized-training-window

python -m sac_qutab compute-preflight --config configs/core.yaml \
  --administrative "$ADMINISTRATIVE_JSON" --splits artifacts/data/splits.json \
  --summary resnet18=artifacts/preflight/resnet18-seed17-mig1/summary.json \
  --summary deit_tiny_patch16_224=artifacts/preflight/deit_tiny_patch16_224-seed17-mig1/summary.json \
  --weight-preflight resnet18=artifacts/preflight/resnet18_weights.json \
  --weight-preflight deit_tiny_patch16_224=artifacts/preflight/deit_tiny_weights.json \
  --output artifacts/preflight/compute-mig1.json
```

Expected artifacts are both `summary.json` files, bounded `latest.pt` checkpoints, and `compute-mig1.json`. Acceptance requires the assigned selector, world size one, A100 capability 8.0, at least 18 GiB visible memory, peak reserved memory ≤18 GiB, positive multiprocessor count, exact pretrained SHA agreement, five warm-up plus twenty timed steps, correct full-train and full-validation projection arithmetic, total projected core work plus allowance ≤36 hours, and ≥12 hours recovery.

If batch 32/accumulation 2 fails by OOM or cap only, retry both summaries with `--max-device-batch 16`, then 8, preserving global batch 64. Compare workers 4 against 8 only in this bounded stage; accept 8 only under the runbook's ≥5% per-model rule. After the selected summaries, resume drill, and v2 preflight pass, launch the canonical single-process command:

```bash
export CUDA_VISIBLE_DEVICES="$SAC_MIG_UUID"
python -m sac_qutab run-all --config configs/core.yaml \
  --pair-manifest validation=artifacts/counterfactuals/validation.jsonl \
  --pair-manifest calibration=artifacts/counterfactuals/calibration.jsonl \
  --pair-manifest final_test=artifacts/counterfactuals/final_test.jsonl \
  --review-freeze artifacts/review/review_freeze.json \
  --administrative "$ADMINISTRATIVE_JSON" \
  --preflight artifacts/preflight/compute-mig1.json \
  --authorized-training-window
```

## Known limitations

- Git cleanliness/diff cannot be proven because the supplied directory is not a Git checkout.
- Local CPU/RTX evidence does not validate the reported MIG UUID/profile, visible memory, multiprocessor count, A100 throughput, worker choice, convergence, hardware resume, or the 48-hour booking.
- Test fixtures verify contracts and numerical behavior, not EuroSAT accuracy, semantic validity, calibration, CKA, failure prevalence, AUROC/AP, or scientific hypotheses.
- RQ3 may legitimately be underpowered for any real model seed; the correct output is then an explicit undefined/underpowered record, not a changed threshold or invented metric.
