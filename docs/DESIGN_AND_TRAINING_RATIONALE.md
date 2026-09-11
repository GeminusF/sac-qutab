# SAC-QUTAB Design and Training Rationale

## Scientific contract

SAC-QUTAB is a Track 1 pure-research study over EuroSAT RGB. The active `sac-campaign-v1` experiment is:

- ResNet-50 and ViT-Small/16 are compared under a shared 224-pixel preprocessing path, with native 64-pixel views retained for intervention construction.
- Seeds are 17, 29, and 43. The grouped train/validation/calibration/final-test fractions are 60/15/10/15.
- The effective global batch is 64. ResNet uses LR 3e-4 and ViT uses LR 5e-4. Per-device batch may be 32, 16, or 8 with matching accumulation, but the accepted value is frozen campaign-wide before full training.
- Validation selects checkpoints and intervention conditions; calibration fits the failure detector; final test is used once under the frozen protocol.
- For the active `sac-campaign-v1` A100 execution, no human semantic-preservation review is performed. All nine predeclared non-sham transform conditions are retained by an authenticated condition-selection policy; synthetic reviewer fixtures are pipeline QA only, not scientific evidence. Results therefore characterize model behavior under engineered transforms and do not establish human-validated semantic preservation.
- RQ1–RQ3, the configured task and counterfactual metrics, paired statistics, and provenance contracts remain the scientific acceptance surface. Optional MAE, multispectral, spatial, and gamma work remains out of scope.

The five-model base Config remains an immutable provenance source; it is not the active execution matrix. Both active implementations use pinned timm backends and exact locally staged ImageNet-1K safetensors checkpoints. ViT-Small/16 (22.1M parameters, 384-d CLS representation) is capacity-near enough to ResNet-50 for a meaningful CNN–Transformer representation comparison, while remaining practical on the 20 GB MIG slice. This makes the RQ2 contrast architectural rather than a comparison against a tiny under-capacity transformer.

Within-model RQ1/RQ2 analyses cover two models. Cross-model representation analysis has one frozen right-minus-left contrast: ResNet-50 → ViT-Small/16. Original-image clustered intervals and the predefined Holm policy remain; with one pair, Holm is mathematically trivial and is reported as such.

## Portable training decisions

The active full-training target is the academy A100 `3g.20gb` MIG allocation. `train.device=cuda` is required for real runs; explicit CPU/auto paths remain only for bounded development checks.

Portable optimizations are retained:

- the campaign hardware gate requires BF16 on the A100 MIG device;
- explicit `bf16`, `fp16`, and `none` policies are available;
- gradient accumulation preserves the configured effective global batch;
- pinned-memory loaders, worker configuration, deterministic seeding, atomic checkpoints, and generic DDP remain supported;
- checkpoints contain model, optimizer, scheduler, scaler, RNG, data/config/weight provenance, optimizer step, and W&B run identity.

The campaign preflight authenticates two measured seed-17 CUDA dry runs, split/weight/config hashes, effective batch, BF16, peak reserved memory below 85%, A100 20 GB visibility, and projected training+validation at no more than three hours per seed. H100 ResNet and A100 ViT training provenance are frozen separately; all final inference uses the same A100 preflight.

## W&B observability

W&B is an optional adapter around the existing local JSONL and checkpoint evidence. Disabled mode has no SDK dependency at runtime. Offline and online modes log resolved run/config metadata, train and validation loss, accuracy, Macro-F1, learning rate, epoch, optimizer step, best checkpoint facts, and final summary values. Dataset and checkpoint artifacts are not uploaded automatically.

Training never performs interactive authentication. Online mode uses `WANDB_API_KEY` or a prior external `wandb login`. A checkpoint's W&B run ID resumes logging continuity, while `--resume` restores the distinct model/optimizer training state. W&B logging errors are recorded without replacing the underlying training exception, and all exit paths finish the run.

## Evidence boundary

Local tests establish contract behavior, CPU-safe end-to-end mechanics, CUDA-preflight validation, and W&B adapter behavior with a no-network SDK double. They do not establish the pending A100 preflight, ViT convergence, or final scientific results. See [Training Readiness](training_readiness.md) and the active [A100 MIG runbook](A100_TRAINING_RUNBOOK.md).
