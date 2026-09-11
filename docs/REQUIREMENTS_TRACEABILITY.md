# SAC-QUTAB Requirements Traceability

`verified locally` means exercised without claiming a real dataset, rented GPU, or scientific result. The assignment and experiment-design sources retain authority for the scientific protocol; their academy-hosting assumptions were superseded on 2026-09-07.

| Requirement | Current implementation | Evidence/status |
|---|---|---|
| One course track and reproducible research question | Track 1; RQ1–RQ3; negative results remain reportable | Config/schema/report contracts; verified locally |
| Fair model comparison | ResNet-50, ConvNeXt-Tiny, ViT-Small/16, Swin-Tiny, DINOv2-Small; common 224 preprocessing; seeds 17/29/43; fixed model-specific learning rates | Config/model/train tests; real checkpoints pending |
| Leakage-safe data protocol | EuroSAT RGB, 27,000 images/10 classes, grouped 60/15/10/15 roles, immutable hashes | Manifest/split tests; real dataset pending |
| Effective optimization settings | Global batch 64; derived accumulation; configured optimizers/schedulers/epochs; portable AMP | Config/train/checkpoint tests; CUDA pending |
| Counterfactual validity | Fixed intervention generation, authenticated provenance cache, and explicit all-predeclared-condition policy | Intervention/cache tests; no human semantic-preservation claim in the active campaign |
| Metrics and inference | Task loss/accuracy/Macro-F1; RQ metrics; detector calibration; five within-model and 10 Holm-corrected pairwise contrasts; single frozen final test | Evaluation/statistics/report tests; trained artifacts pending |
| Portable NVIDIA execution | Explicit CUDA default, configurable device/batch/workers/paths/precision, generic DDP, measured hardware preflight | Governance/CLI tests; rented GPU pending |
| No academy coupling | No administrative/MIG/A100 allowlist, fixed server path, academy memory/SM/window gate, or team access dependency | Source/config/CLI scan; verified locally |
| Checkpoint and resume | Atomic latest/best state with optimizer, scheduler, scaler, RNG, provenance, optimizer step, and W&B ID | Train tests and synthetic smoke |
| W&B | Disabled/offline/online; external auth; resolved model/pretraining/weight/adapter identity; train/validation axes; best/final summary; cleanup | Adapter/train tests; actual instance offline check pending |
| Reproducible operations | Exact migration, A100 preflight, prepare/finalize, resume, and export commands | [A100 MIG runbook](A100_TRAINING_RUNBOOK.md); locally checked CLI surface |
| Submission outputs | Required tables/plots/checkpoints/results and honest evidence boundary | Reporting code exists; actual execution and submission remain manual |

The compact current acceptance record is [Training Readiness](training_readiness.md). Historical closure reports are evidence of earlier implementation work, not current academy launch instructions.
