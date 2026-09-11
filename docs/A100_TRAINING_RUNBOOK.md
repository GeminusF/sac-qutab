# A100 MIG two-model campaign runbook

This is the active operational path for `sac-campaign-v1`: ResNet-50 and ViT-Small/16, each at seeds 17, 29, and 43. The frozen five-model `configs/core.yaml` remains unchanged. The campaign applies the original `/workspace` and `data.num_workers=8` overrides, so the effective base digest remains `cc73206d4ff490bebaa9440af5689decd4a93d08038df486dadd8dfe0deb449c` and completed ResNet checkpoints retain their identity.

## 1. Build and upload the continuation bundle

On Windows PowerShell, from the repository:

```powershell
.\scripts\build_continuation_bundle.ps1 `
  -Archive "C:\Users\fereh\Downloads\sac-qutab-backup\sac-qutab-full-migration-backup.tar" `
  -OutputDirectory ".\migration-output"
```

The builder rechecks the fixed 14,067,722,240-byte source archive and SHA-256 `cb9ef494946f4ca43a883768904df234fd2f57b36e46c3cdc61f29c0957337f2`, rejects path traversal, extracts only the continuation allowlist, overlays this campaign-aware code, writes a per-file SHA-256 manifest, and emits 512 MiB parts. It excludes the counterfactual tensor cache, unused model weights/runs, W&B caches, `.env` files, and secret-like WireGuard/API-key files.

Upload every `sac-qutab-continuation.tar.partNNN` and `sac-qutab-continuation.tar.sha256` file through JupyterLab. Use a JupyterLab Terminal for all commands; do not run long training in a notebook cell.

## 2. Restore and verify

```bash
chmod +x scripts/restore_continuation_bundle.sh scripts/package_campaign_export.sh
bash scripts/restore_continuation_bundle.sh /path/to/uploaded-parts
cd /workspace/sac-qutab
export PYTHONPATH=src
```

If `/workspace` is unavailable to the Jupyter user, pass a writable physical root as the second argument:

```bash
bash /path/to/uploaded-parts/restore_continuation_bundle.sh /path/to/uploaded-parts /sdb-disk/notebooks/team10/workspace
source /sdb-disk/notebooks/team10/workspace/sac-qutab-runtime.env
cd "$SAC_QUTAB_PHYSICAL_ROOT/sac-qutab"
```

The restore records a path-relocation sidecar without rewriting checkpoint/config metadata. It verifies the bundle and every restored file, then records `df`, `nvidia-smi`, Python, pip, and Torch/CUDA evidence under the physical artifact root.

Confirm the visible allocation is A100 MIG `3g.20gb` and the A100 CUDA 12.4 lock is present. The legacy H100 training records retain their original Torch 2.13/CUDA 12.6 metadata; they are not rewritten:

```bash
nvidia-smi -L
python -c "import torch,timm,safetensors,wandb,torchvision; print(torch.__version__,torchvision.__version__,timm.__version__,safetensors.__version__,wandb.__version__); assert torch.__version__ == '2.6.0+cu124'; assert torchvision.__version__ == '0.21.0+cu124'; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); assert 'A100' in torch.cuda.get_device_name(0).upper()"
python -m sac_qutab validate-config --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --world-size 1
```

The observed academy allocation uses driver 550.127.08 and reports CUDA 12.4. Its frozen execution stack is Python 3.12, Torch 2.6.0+cu124, torchvision 0.21.0+cu124, timm 1.0.29, safetensors 0.8.0, and W&B 0.29.0, as locked in `requirements-a100-cu124.txt`. All three H100 ResNet checkpoints must pass the cross-version load gate before they may be reused. Do not silently substitute versions during the experiment window; stop and reconcile the environment first.

When application-level relocation is active, export both roots before invoking the CLI. Hash serialization maps the physical root back to `/workspace`, so the base Config and completed ResNet resolved-run identities remain unchanged:

```bash
export SAC_QUTAB_LOGICAL_ROOT=/workspace
export SAC_QUTAB_PHYSICAL_ROOT=/sdb-disk/notebooks/team10/workspace
cd "$SAC_QUTAB_PHYSICAL_ROOT/sac-qutab"
export PYTHONPATH="$PWD/src"
```

The exact 5.95 GiB counterfactual tensor cache must be transferred from the authenticated source archive. Do not regenerate it under a different Torch build: Torch 2.6/CUDA 12.4 reproduced the row structure but changed 28,350 texture/resolution cache hashes relative to the Torch 2.13 source. Build the dedicated 13-part cache bundle with `scripts/build_cache_transfer_bundle.ps1`, authenticate the combined archive SHA-256, extract into an isolated staging directory, and verify all 129,606 files against `cache-transfer-files.sha256` before activating it.

After activation, validate the three roles without regenerating them:

```bash
for role in validation calibration final_test; do
  python -m sac_qutab cache-validate --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --pairs /workspace/artifacts/counterfactuals/$role.jsonl
done
```

The authenticated original manifest SHA-256 values are `0fc0f2724d149825ad9bbe400a0256008ae18d0a745a50a7ef206bcaec9238b5` (validation), `e5fb1491c381ce32af663860fb1f356234a55f361a8a3a56e042cfcff8da77f5` (calibration), and `10cb69bbcd74680af82e1ea10e2901e0c4a02270773b34b5c9f7648cd94b0308` (final test). Preserve any cross-version regeneration attempt as diagnostic evidence; do not alter a frozen pair manifest to accommodate a cache mismatch.

## 3. W&B without persistent secrets

Enter the API key only in the terminal environment. Do not put it in a notebook, bundle, shell-history file, or repository:

```bash
read -s -p 'WANDB API key: ' WANDB_API_KEY; export WANDB_API_KEY; echo
export WANDB_MODE=online
export WANDB_PROJECT=sac-qutab
export WANDB_RUN_GROUP=sac-campaign-v1-a100-mig
export SAC_EXECUTION_PROVIDER=academy-a100-mig
```

Local JSON, JSONL, checkpoints, and SHA-256 files remain authoritative.

## 4. A100 bounded preflight

Authenticate weights and run 5 warm-up plus 20 measured optimizer steps, starting at physical batch 32:

```bash
python -m sac_qutab validate-weights --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --model resnet50 --checkpoint /workspace/weights/resnet50.tv_in1k.safetensors > /workspace/artifacts/preflight/resnet50_weights.json
python -m sac_qutab validate-weights --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --model vit_small_patch16_224 --checkpoint /workspace/weights/vit_small_patch16_224.augreg_in1k.safetensors > /workspace/artifacts/preflight/vit_small_patch16_224_weights.json
python -m sac_qutab train --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --model resnet50 --seed 17 --run-dir /workspace/artifacts/preflight/a100_resnet50 --dry-run --max-device-batch 32 --weight-preflight /workspace/artifacts/preflight/resnet50_weights.json
python -m sac_qutab train --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --model vit_small_patch16_224 --seed 17 --run-dir /workspace/artifacts/preflight/a100_vit_small --dry-run --max-device-batch 32 --weight-preflight /workspace/artifacts/preflight/vit_small_patch16_224_weights.json
```

If ViT OOMs or exceeds the memory gate, retry only at `--max-device-batch 16`, then `8`. Update only `max_device_batch.vit_small_patch16_224` in the campaign file to the accepted value before creating the campaign preflight. This changes the campaign SHA-256 but not the base Config or ResNet resolved hash. Never reduce epoch count or switch model automatically.

```bash
python -m sac_qutab hardware-preflight --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml \
  --splits /workspace/artifacts/data/splits.json \
  --summary resnet50=/workspace/artifacts/preflight/a100_resnet50/summary.json \
  --summary vit_small_patch16_224=/workspace/artifacts/preflight/a100_vit_small/summary.json \
  --weight-preflight resnet50=/workspace/artifacts/preflight/resnet50_weights.json \
  --weight-preflight vit_small_patch16_224=/workspace/artifacts/preflight/vit_small_patch16_224_weights.json \
  --output /workspace/artifacts/preflight/a100-campaign.json
```

This fails unless CUDA/BF16 timing is valid, peak reserved VRAM is below 85%, hashes match, and projected training plus validation is at most 3 hours per seed. Also manually confirm `nvidia-smi -L` reports the expected `3g.20gb` administrative profile.

## 5. Prepare without final-test access

No human semantic-preservation review is part of this campaign. The historical synthetic reviewer/adjudicator files are test-fixture output only and must not be cited as human evidence. Freeze an explicit all-predeclared-condition policy bound to the authenticated validation manifest:

```bash
python -m sac_qutab condition-policy --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml \
  --pairs /workspace/artifacts/counterfactuals/validation.jsonl \
  --output /workspace/artifacts/review/no_human_condition_policy.json
```

This retains all nine predeclared non-sham family×severity conditions by design. It makes no semantic-preservation claim, records `human_semantic_review_performed: false`, and treats the synthetic review fixture as non-scientific pipeline QA.

```bash
python -m sac_qutab run-all --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --stage prepare \
  --pair-manifest validation=/workspace/artifacts/counterfactuals/validation.jsonl \
  --pair-manifest calibration=/workspace/artifacts/counterfactuals/calibration.jsonl \
  --condition-policy /workspace/artifacts/review/no_human_condition_policy.json \
  --training-preflight resnet50=/workspace/artifacts/preflight/hardware.json \
  --training-preflight vit_small_patch16_224=/workspace/artifacts/preflight/a100-campaign.json
```

The legacy H100 preflight is used only for ResNet training provenance. Existing ResNet training, validation, calibration, and detector files are reused without rewrite. ViT seeds run in order 17, 29, 43. `latest.pt` is atomically saved every 100 optimizer steps and at epoch end. Resume only the exact latest checkpoint:

```bash
python -m sac_qutab train --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --model vit_small_patch16_224 --seed 17 --run-dir /workspace/artifacts/runs/vit_small_patch16_224-seed17 --resume /workspace/artifacts/runs/vit_small_patch16_224-seed17/checkpoints/latest.pt --preflight /workspace/artifacts/preflight/a100-campaign.json
bash scripts/package_campaign_export.sh incremental 17 /workspace/exports
```

Repeat the incremental export for seeds 29 and 43 and verify each locally.

## 6. Freeze and final test

Only after all six checkpoints, validation ECDFs, calibration evaluations, and detectors exist:

```bash
python -m sac_qutab run-all --config configs/core.yaml --campaign configs/sac-campaign-v1.yaml --stage finalize \
  --pair-manifest validation=/workspace/artifacts/counterfactuals/validation.jsonl \
  --pair-manifest calibration=/workspace/artifacts/counterfactuals/calibration.jsonl \
  --pair-manifest final_test=/workspace/artifacts/counterfactuals/final_test.jsonl \
  --condition-policy /workspace/artifacts/review/no_human_condition_policy.json \
  --training-preflight resnet50=/workspace/artifacts/preflight/hardware.json \
  --training-preflight vit_small_patch16_224=/workspace/artifacts/preflight/a100-campaign.json \
  --inference-preflight /workspace/artifacts/preflight/a100-campaign.json
```

`finalize` authenticates all preparation artifacts before protocol v5, then evaluates all six runs in the common A100 environment, evaluates frozen detectors, generates primary statistics, and builds the report. Existing final-test outputs are never silently overwritten.

## 7. Final export and deletion gate

```bash
bash scripts/package_campaign_export.sh final ignored /workspace/exports
```

Reassemble downloaded parts locally, verify archive SHA-256, extract into a new directory, and run `sha256sum -c files.sha256`. Do not delete A100 material until archive and per-file verification both succeed locally.

Acceptance requires 3 ResNet + 3 ViT best checkpoints; 6 validation, calibration, detector-fit, and final-test outputs; one v5 protocol; primary statistics/report; and a final archive verified on both machines.

## 8. Completed campaign reproducibility record

The two-model campaign completed on 2026-09-09. Treat the following identifiers as the immutable provenance record for analysis and reporting:

| Item | Frozen value |
|---|---|
| Active run matrix | ResNet-50 and ViT-Small/16, seeds 17, 29, 43 (6 runs) |
| Effective base configuration SHA-256 | `cc73206d4ff490bebaa9440af5689decd4a93d08038df486dadd8dfe0deb449c` |
| Campaign SHA-256 | `d6dd3cabca656ef4414b165d82de1d9f45a228418a25e84e696eab9f248c7e8b` |
| No-human condition-policy SHA-256 | `18838e7446d37c172a3fc7cf686779bf7c4309612378c836261ad0deaae31d2a` |
| Authenticated final-test manifest SHA-256 | `10cb69bbcd74680af82e1ea10e2901e0c4a02270773b34b5c9f7648cd94b0308` |
| Frozen protocol SHA-256 | `cd20ca70be0619e87b2422fa99f7601c9b86578e06475f15a385903c9ed134a9` |
| Final export archive SHA-256 | `8aa20619ea4d3865c7c44ec68f7007ba1d3225808853a7e0c27b96c684bc5ac1` |

The final pipeline emitted `FINALIZE_COMPLETED` and was independently checked as `FINAL_PIPELINE_FULLY_VERIFIED`: all 6 runs were present, the report contained 68 generated files (47 PNG figures and 15 tables), and every evaluation index and detector-metrics file matched the frozen protocol. The final export was verified on the A100 host and again after local download. The local extraction check reported `LOCAL_FINAL_BACKUP_FULLY_VERIFIED` with 424 files authenticated against the internal `files.sha256` manifest.

The canonical local recovery copies are:

```text
C:\Users\fereh\Downloads\sac-qutab-final-backup\sac-qutab-two-model-final-20260909T084149Z.tar
C:\Users\fereh\Downloads\sac-qutab-final-backup\verified-extract
```

On Windows PowerShell, recheck the archive identity without modifying it:

```powershell
$archive = "C:\Users\fereh\Downloads\sac-qutab-final-backup\sac-qutab-two-model-final-20260909T084149Z.tar"
$expected = "8aa20619ea4d3865c7c44ec68f7007ba1d3225808853a7e0c27b96c684bc5ac1"
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Final archive SHA-256 mismatch" }
Write-Host "FINAL_ARCHIVE_VERIFIED=$actual"
```

Use `verified-extract/artifacts/report_artifacts/final` as the source for scientific tables and figures, and use W&B as a synchronized viewing copy rather than the sole source of truth. Do not rerun `finalize` against the same output directories or reinterpret synthetic pipeline QA as human semantic review. A complete reproduction must use the pinned environment above, the exact three manifests and policy, the six-run matrix, and the same freeze-before-final-test sequence.
