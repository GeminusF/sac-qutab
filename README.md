# SAC-QUTAB

A frozen six-run EuroSAT RGB study of prediction, representation, calibration and failure detection: ResNet-50 and ViT-Small/16, seeds 17, 29 and 43.

## Install and reproduce headline numbers

Use source release `v1.0.1-final` (or current `main`) and Python 3.12 in a fresh environment. Repository attributes preserve the hash-pinned file bytes on Windows, Linux and macOS:

```bash
git clone --branch v1.0.1-final https://github.com/GeminusF/sac-qutab.git
cd sac-qutab
```

Create the environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-ci.txt
```

After setup, one command downloads the hash-pinned numeric evidence and recomputes the headline estimates:

```bash
python scripts/reproduce.py --mode headline
```

The evidence URL targets release `v1.0-final`. Until that release is published, use `--evidence /path/to/all_artifacts` with the original frozen extraction. A missing release fails explicitly; it never substitutes synthetic results.

| Model | Clean accuracy | Severe-resolution accuracy | Mean detector AP gain |
|---|---:|---:|---:|
| ResNet-50 | 98.63% | 12.94% | 0.432 |
| ViT-Small/16 | 96.77% | 43.00% | 0.402 |

The replay recalculates accuracy from saved probabilities and detector AP from calibrated coefficients. It does not train, run neural inference, or re-estimate bootstrap intervals. Output is written to `output/headline.json`. The ~134 MiB compressed evidence contains authenticated numeric outputs, not raw images or checkpoints.

## Tests

```bash
# PowerShell: $env:PYTHONPATH='src'
# Linux/macOS: export PYTHONPATH=src
python -m sac_qutab validate-config --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --world-size 1
python -m pytest -q -p no:cacheprovider
python scripts/check_submission.py
```

CI runs CPU checks only. GPU training and deployment are not part of CI.

## Full experiment and report

- [Data and exact weights](docs/DATA_AND_WEIGHTS.md)
- [Environment profiles](docs/ENVIRONMENT.md)
- [Prepare/finalize runbook](docs/A100_TRAINING_RUNBOOK.md)
- [Frozen completion evidence](docs/FINAL_EVIDENCE.md)
- [Scientific rationale](docs/DESIGN_AND_TRAINING_RATIONALE.md)
- [Report source and build](report/README.md)
- [Presentation](presentation/presentation.pdf)
- [Publication provenance](docs/PUBLICATION.md)
- [Contribution and review policy](docs/REVIEW_POLICY.md)

Report compilation uses retained `report/figures/` assets and needs no external materials folder. For a full numeric report audit, set `SAC_EVIDENCE_ROOT` to your authenticated full extraction. The original `prepare â†’ freeze â†’ finalize` training protocol and all frozen scientific identifiers remain unchanged.

## Limitations and contribution record

There was no human semantic-preservation review. Model pretraining and training hardware differ; training throughput is not a controlled architecture comparison. Seed and domain coverage are limited. Synthetic tests establish wiring, not scientific conclusions.

This repository is a publication of an already completed study. The contribution PDF records the equal publication assignment agreed by all five members, as confirmed by the team representative. Actual GitHub activity is obtained with `python scripts/audit_github.py`; it is not inferred from the assignment. The local account-switching controller is intentionally outside the repository.
