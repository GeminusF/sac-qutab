# Training Readiness

Date: 2026-09-08

## Verdict

**Codebase: READY for the documented two-model A100 preflight.** Campaign selection is external to the frozen Config, preserves the completed ResNet resolved hashes, supports mixed H100/A100 training provenance, separates prepare from finalize, and integrates optional W&B logging without weakening local evidence.

**Academy A100 MIG execution: NOT YET VERIFIED.** The user-reported `3g.20gb` allocation is visible in supplied evidence, but the new two-model bounded preflight, ViT training, and common-session final inference have not yet run.

## Migration and verification scope

- Retained the five-model Config byte-for-byte as legacy provenance and added a separately hashed two-model campaign.
- Retained global batch 64, model learning rates, seeds, splits, objectives, metrics, AMP, gradient accumulation, pinned-memory loading, deterministic state, checkpointing, generic DDP, and protocol authentication.
- Added A100 20 GB MIG/BF16, 85% memory, and three-hour-per-seed campaign gates while preserving legacy generic preflight v1.
- Added W&B disabled/offline/online modes, external authentication, resolved run metadata, consistent optimizer-step metrics, best/final summaries, exception-safe cleanup, and checkpoint-bound W&B run identity.
- Added exact migration, A100 preflight, prepare/finalize, resume, and 512 MiB export steps in [A100_TRAINING_RUNBOOK.md](A100_TRAINING_RUNBOOK.md).

## Verification evidence

| Check | Result |
|---|---|
| Python compilation | passed: `python -m compileall -q src tests` |
| Frozen legacy Config | passed: file SHA-256 `139c544a...e8ec`; effective `/workspace` Config SHA-256 `cc73206d...449c` |
| Campaign matrix | passed: exactly ResNet-50 + ViT-Small/16 × seeds 17/29/43 and one ordered pair |
| Complete CPU-safe test suite | passed: 57 tests in 105.54 s; 90 expected warnings from intentionally small synthetic fixtures |
| Backup source | passed: 14,067,722,240 bytes and SHA-256 `cb9ef494946f4ca43a883768904df234fd2f57b36e46c3cdc61f29c0957337f2` |
| Bounded synthetic forward/backward, optimizer, validation, checkpoint/load/evaluation | passed in about 31 s under `artifacts/local-smoke-vast`; W&B disabled |
| `.omp/` ignored, absent from index, local files preserved | passed; only the seven formerly tracked paths are staged for index removal |
| Active-code academy-reference scan and diff whitespace check | passed; only explicit historical/supersession documentation mentions remain; no whitespace errors |

W&B integration tests use a deterministic no-network SDK double. The local environment does not currently contain the newly declared `wandb` dependency, so a real SDK offline directory was not created. Package installation was deliberately not performed during this task.

## Remaining required gates

1. **Build/upload/restore the continuation bundle.** Run the allowlist builder and verify archive plus per-file hashes under logical `/workspace`.
2. **Run the A100 campaign preflight.** Measure ResNet and ViT at seed 17; accept batch 32, or ViT 16/8 only after campaign update; require BF16, <85% VRAM, and ≤3 hours per seed.
3. **Run `prepare`.** Reuse the three ResNet runs without rewrite and complete three ViT runs plus validation, calibration, and detector-fit. Export each ViT seed incrementally.
4. **Run `finalize`.** Freeze all six inputs with mixed training provenance and common A100 inference evidence before any final-test scoring.
5. **Verify the final export locally.** Do not delete A100 material until both archive SHA-256 and every file digest pass.
