# Historical five-model implementation plan

> Superseded operationally on 2026-09-08 by `configs/sac-campaign-v1.yaml` and [A100_TRAINING_RUNBOOK.md](A100_TRAINING_RUNBOOK.md). Retained only as legacy evidence.

| Milestone | Acceptance criterion | Status |
|---|---|---|
| Academic and scientific contract | Track, models, seeds, split roles, preprocessing, objectives, metrics, review, and final-test discipline agree with the source documents | complete |
| Data and intervention pipeline | Deterministic content manifest/grouped split, leakage checks, paired interventions, cache provenance, and blinded review/freeze are implemented | complete locally; real inputs pending |
| Models and optimization | Five pinned timm models, exact full-SHA safetensors authentication, effective batch 64, fixed per-model LR, losses, schedulers, AMP, validation, and checkpoint/resume are connected | complete in code/tests; real timm weights and CUDA pending |
| Academy decoupling | No active A100/MIG identity, team access record, fixed device profile, academy budget/window, or device-family refusal remains | complete |
| Vast.ai portability | Configurable paths/device/batch/workers/precision, clear CUDA failure, measured generic hardware preflight, smoke/full/resume/export commands | complete in code/docs; instance preflight pending |
| W&B integration | Disabled/offline/online modes, environment/config identity, scalar axes, summaries, failure cleanup, checkpoint run ID, no automatic data upload | complete in code/tests; real SDK offline run pending |
| Local verification | Focused regression tests, full CPU-safe suite, bounded synthetic train/validation/checkpoint-load, diff and ignore audits pass | complete |
| Statistical expansion | Five within-model analyses plus all 10 ordered right-minus-left model pairs, clustered intervals, Holm primary correction, protocol v4/statistics v3 | complete in code/tests |
| Scientific execution | Real EuroSAT and five exact weights pass validation, review freeze exists, five CUDA dry runs pass hardware preflight, 15 runs and frozen evaluation complete | blocked on manual/rented-instance inputs |

The exact operator sequence and completion criteria are in [Vast.ai GPU Training Guide](training_vast_ai.md). Verification results and only material remaining gates are in [Training Readiness](training_readiness.md).
