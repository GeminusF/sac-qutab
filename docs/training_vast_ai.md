# Historical Vast.ai GPU Training Guide

This file preserves the superseded five-model rental workflow for provenance. It is not the active campaign guide. Use [A100_TRAINING_RUNBOOK.md](A100_TRAINING_RUNBOOK.md) for the two-model academy A100 MIG workflow; do not copy commands from this historical document into the active campaign.

## 1. Select, transfer, and prepare

Choose a Linux/NVIDIA image and persistent disk large enough for EuroSAT, five weight files, cached counterfactuals, 15 run directories, and an export archive. Do not assume the GPU model, VRAM, CUDA version, throughput, worker count, or multi-GPU topology before inspecting the instance. Follow Vast.ai's [official quickstart](https://docs.vast.ai/guides/get-started/quickstart), and export all required files before deleting the rental.

Placeholders below must be replaced:

    rsync -av --progress -e 'ssh -p <PORT>' ./ '<USER>@<HOST>:/workspace/sac-qutab/'
    ssh -p <PORT> '<USER>@<HOST>'
    cd /workspace/sac-qutab

Create an environment. Select the official PyTorch wheel/index compatible with the instance driver and CUDA stack; do not guess it before nvidia-smi:

    nvidia-smi
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    # If the image does not already supply the compatible pinned build:
    python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url '<PYTORCH_INDEX_URL>'
    python -m pip install -e .

Record and check the environment:

    mkdir -p /workspace/artifacts/preflight
    nvidia-smi | tee /workspace/artifacts/preflight/nvidia-smi.txt
    python -c "import torch, torchvision, timm, safetensors, wandb; print(torch.__version__, torchvision.__version__, timm.__version__, safetensors.__version__, wandb.__version__); print(torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_properties(0))" | tee /workspace/artifacts/preflight/environment.txt
    python -m pip freeze > /workspace/artifacts/preflight/pip-freeze.txt
    python -m sac_qutab validate-config --config configs/core.yaml --world-size 1 > /workspace/artifacts/preflight/config.json

Completion criterion: all pinned dependencies import, one intended CUDA device is visible, and validate-config emits exactly 15 runs with effective global batch 64.

## 2. Portable path and hardware settings

Use one consistent override set for the whole campaign:

    COMMON=(
      --set 'data.root=/workspace/data/eurosat'
      --set 'project.output_dir=/workspace/artifacts'
      --set 'data.num_workers=8'
      --set 'train.device=cuda'
      --set 'train.amp_dtype=auto'
    )

The standard path is one GPU selected through CUDA_VISIBLE_DEVICES. The lower-level trainer supports generic torchrun/NCCL, but run-all is deliberately sequential and single-process.

- --max-device-batch controls physical batch; accumulation preserves effective global batch 64.
- data.num_workers is instance-specific.
- auto precision chooses BF16 only if supported and otherwise FP16 with GradScaler.
- A requested CUDA run fails instead of silently using CPU.
- Use the same physical-batch choice for every model's smoke, hardware freeze, full run, and resume.

If any model OOMs, retry all five smoke runs with a smaller common value such as 16 or 8. Do not alter effective batch, learning rate, seeds, input size, or precision policy merely to hide an OOM.

## 3. Data and counterfactual inputs

Download EuroSAT on the Vast.ai instance to durable storage. The temporary archive and extraction directory make an interrupted or malformed download distinct from the accepted dataset path:

    mkdir -p /workspace/data/eurosat-extracted
    curl --fail --location --retry 3 'https://madm.dfki.de/files/sentinel/EuroSAT.zip' --output /workspace/data/EuroSAT.zip.part
    mv /workspace/data/EuroSAT.zip.part /workspace/data/EuroSAT.zip
    python -m zipfile -e /workspace/data/EuroSAT.zip /workspace/data/eurosat-extracted
    test -d /workspace/data/eurosat-extracted/2750
    test ! -e /workspace/data/eurosat
    mv /workspace/data/eurosat-extracted/2750 /workspace/data/eurosat

If the canonical source is unavailable, transfer the archive or already extracted tree to the same durable path. Do not accept it by filename alone: the repository manifest and split commands below authenticate the complete image set. The resulting 27,000 EuroSAT RGB images must be under ten class directories:

    /workspace/data/eurosat/
      AnnualCrop/ Forest/ HerbaceousVegetation/ Highway/ Industrial/
      Pasture/ PermanentCrop/ Residential/ River/ SeaLake/

Build and validate the immutable grouped split:

    python -m sac_qutab data-manifest --config configs/core.yaml "${COMMON[@]}"
    python -m sac_qutab split-create --config configs/core.yaml "${COMMON[@]}"
    python -m sac_qutab data-verify --config configs/core.yaml "${COMMON[@]}"

Create the shared validation, calibration, and final-test counterfactual manifests:

    for ROLE in validation calibration final_test; do
      python -m sac_qutab pairs-create --config configs/core.yaml "${COMMON[@]}" --role "$ROLE" --output "/workspace/artifacts/counterfactuals/$ROLE.jsonl"
      python -m sac_qutab cache-generate --config configs/core.yaml "${COMMON[@]}" --pairs "/workspace/artifacts/counterfactuals/$ROLE.jsonl"
      python -m sac_qutab cache-validate --config configs/core.yaml "${COMMON[@]}" --pairs "/workspace/artifacts/counterfactuals/$ROLE.jsonl"
    done

Export the blinded validation-only review package:

    python -m sac_qutab review-export --config configs/core.yaml "${COMMON[@]}" --pairs /workspace/artifacts/counterfactuals/validation.jsonl --output-dir /workspace/artifacts/review

Two qualified reviewers complete their files independently; a distinct adjudicator resolves disagreements. Replace the attestation placeholder:

    python -m sac_qutab review-freeze --config configs/core.yaml "${COMMON[@]}" --review-dir /workspace/artifacts/review --pairs /workspace/artifacts/counterfactuals/validation.jsonl --attestation '<REVIEW_ATTESTATION_JSON>' --output /workspace/artifacts/review/review_freeze.json

Completion criterion: data-verify reports the configured counts and review-freeze succeeds without inspecting final-test model outputs.

## 4. Exact pretrained weights

Create a durable weights directory and obtain the exact files declared by configs/core.yaml. Download to temporary names so a partial transfer is never mistaken for an accepted checkpoint; transferred files are equally valid:

    mkdir -p /workspace/weights
    curl --fail --location --retry 3 'https://huggingface.co/timm/resnet50.tv_in1k/resolve/main/model.safetensors' --output /workspace/weights/resnet50.tv_in1k.safetensors.part
    curl --fail --location --retry 3 'https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/main/model.safetensors' --output /workspace/weights/convnext_tiny.fb_in1k.safetensors.part
    curl --fail --location --retry 3 'https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k/resolve/main/model.safetensors' --output /workspace/weights/vit_small_patch16_224.augreg_in1k.safetensors.part
    curl --fail --location --retry 3 'https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k/resolve/main/model.safetensors' --output /workspace/weights/swin_tiny_patch4_window7_224.ms_in1k.safetensors.part
    curl --fail --location --retry 3 'https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/main/model.safetensors' --output /workspace/weights/vit_small_patch14_dinov2.lvd142m.safetensors.part
    printf '%s  %s\n' \
      '5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888' '/workspace/weights/resnet50.tv_in1k.safetensors.part' \
      '08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276' '/workspace/weights/convnext_tiny.fb_in1k.safetensors.part' \
      'c20a91d93f4757b0f581519f21ad91a875cf041764873a7a5bccc8e36f360dbf' '/workspace/weights/vit_small_patch16_224.augreg_in1k.safetensors.part' \
      'fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1' '/workspace/weights/swin_tiny_patch4_window7_224.ms_in1k.safetensors.part' \
      '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081' '/workspace/weights/vit_small_patch14_dinov2.lvd142m.safetensors.part' | sha256sum --check
    for FILE in /workspace/weights/*.safetensors.part; do mv "$FILE" "${FILE%.part}"; done

Authenticate and structurally load every file:

    python -m sac_qutab validate-weights --config configs/core.yaml "${COMMON[@]}" --model resnet50 --checkpoint /workspace/weights/resnet50.tv_in1k.safetensors > /workspace/artifacts/preflight/resnet50_weights.json
    python -m sac_qutab validate-weights --config configs/core.yaml "${COMMON[@]}" --model convnext_tiny --checkpoint /workspace/weights/convnext_tiny.fb_in1k.safetensors > /workspace/artifacts/preflight/convnext_tiny_weights.json
    python -m sac_qutab validate-weights --config configs/core.yaml "${COMMON[@]}" --model vit_small_patch16_224 --checkpoint /workspace/weights/vit_small_patch16_224.augreg_in1k.safetensors > /workspace/artifacts/preflight/vit_small_patch16_224_weights.json
    python -m sac_qutab validate-weights --config configs/core.yaml "${COMMON[@]}" --model swin_tiny_patch4_window7_224 --checkpoint /workspace/weights/swin_tiny_patch4_window7_224.ms_in1k.safetensors > /workspace/artifacts/preflight/swin_tiny_patch4_window7_224_weights.json
    python -m sac_qutab validate-weights --config configs/core.yaml "${COMMON[@]}" --model dinov2_vits14 --checkpoint /workspace/weights/vit_small_patch14_dinov2.lvd142m.safetensors > /workspace/artifacts/preflight/dinov2_vits14_weights.json

Training never downloads weights. Every later stage rechecks the full SHA-256. DINOv2-Small uses the CC-BY-NC-4.0 checkpoint licence recorded in its [official timm configuration](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/main/config.json).

## 5. W&B

Authentication happens before training, following the official [quickstart](https://docs.wandb.ai/models/quickstart). Do not put credentials in YAML or source code.

No-upload smoke:

    export WANDB_MODE=offline
    export WANDB_PROJECT=sac-qutab
    export WANDB_RUN_GROUP=vast-five-model-preflight

Online runs:

    export WANDB_API_KEY='<secret>'
    export WANDB_MODE=online
    export WANDB_PROJECT=sac-qutab
    export WANDB_ENTITY='<optional-team-or-user>'
    export WANDB_RUN_GROUP=vast-five-model-core

WANDB_MODE=disabled keeps local events, summaries, and checkpoints. W&B records resolved model/weight/pretraining/adapter metadata, seed, optimizer-step train loss and LR, validation loss/accuracy/Macro-F1, precision, hardware/runtime summaries, best checkpoint, and final summary. Evaluation, detector, review-freeze, protocol, statistics, and reporting stages create hash-linked W&B stage runs. The report stage publishes every generated CSV as a W&B Table and every generated PNG as a W&B Image, including learning curves, master-summary data, the 10-pair forest plot, cross-model CKA, throughput-versus-robustness, failure gallery, attention rollout, macro-cluster confusion, calibration/reliability, class stability, and detector figures. The complete local report bundle is also registered as a W&B Artifact. It does not upload the dataset, raw representation arrays, or every checkpoint.

Vast.ai is the execution and durable-storage environment; W&B is the remote experiment view. Local JSON/CSV/PNG/checkpoint files remain the scientific source of truth, and their hashes connect the Vast outputs to the W&B stage runs. Curves become available during training; aggregate five-model plots appear only after the required evaluations/statistics/report stage has enough frozen inputs to compute them.

For curves, open the W&B project and use optimizer_step as the x-axis for train/* and validation/*. Resume follows W&B's [run-resume guidance](https://docs.wandb.ai/models/runs/resuming): the checkpoint stores the W&B run ID, while the separate --resume argument restores model, optimizer, scheduler, scaler, RNG, and training step.

## 6. Five bounded CUDA smoke runs

The bound is 1,600 training samples, 25 optimizer steps (five warm-up plus twenty timed), and one bounded validation pass per model. Before running, set a common candidate batch:

    export CUDA_VISIBLE_DEVICES=0
    export WANDB_MODE=offline
    BATCH=32

Run all five:

    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model resnet50 --seed 17 --run-dir /workspace/artifacts/preflight/resnet50 --weight-preflight /workspace/artifacts/preflight/resnet50_weights.json --dry-run --max-device-batch "$BATCH"
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model convnext_tiny --seed 17 --run-dir /workspace/artifacts/preflight/convnext_tiny --weight-preflight /workspace/artifacts/preflight/convnext_tiny_weights.json --dry-run --max-device-batch "$BATCH"
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model vit_small_patch16_224 --seed 17 --run-dir /workspace/artifacts/preflight/vit_small_patch16_224 --weight-preflight /workspace/artifacts/preflight/vit_small_patch16_224_weights.json --dry-run --max-device-batch "$BATCH"
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model swin_tiny_patch4_window7_224 --seed 17 --run-dir /workspace/artifacts/preflight/swin_tiny_patch4_window7_224 --weight-preflight /workspace/artifacts/preflight/swin_tiny_patch4_window7_224_weights.json --dry-run --max-device-batch "$BATCH"
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model dinov2_vits14 --seed 17 --run-dir /workspace/artifacts/preflight/dinov2_vits14 --weight-preflight /workspace/artifacts/preflight/dinov2_vits14_weights.json --dry-run --max-device-batch "$BATCH"

Inspect and test exact resume on at least one bounded checkpoint:

    python -m sac_qutab inspect-checkpoint --checkpoint /workspace/artifacts/preflight/resnet50/checkpoints/latest.pt
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model resnet50 --seed 17 --run-dir /workspace/artifacts/preflight/resnet50 --resume /workspace/artifacts/preflight/resnet50/checkpoints/latest.pt --weight-preflight /workspace/artifacts/preflight/resnet50_weights.json --dry-run --max-device-batch "$BATCH"

Freeze the measurements:

    python -m sac_qutab hardware-preflight --config configs/core.yaml "${COMMON[@]}" --splits /workspace/artifacts/data/splits.json \
      --summary resnet50=/workspace/artifacts/preflight/resnet50/summary.json \
      --summary convnext_tiny=/workspace/artifacts/preflight/convnext_tiny/summary.json \
      --summary vit_small_patch16_224=/workspace/artifacts/preflight/vit_small_patch16_224/summary.json \
      --summary swin_tiny_patch4_window7_224=/workspace/artifacts/preflight/swin_tiny_patch4_window7_224/summary.json \
      --summary dinov2_vits14=/workspace/artifacts/preflight/dinov2_vits14/summary.json \
      --weight-preflight resnet50=/workspace/artifacts/preflight/resnet50_weights.json \
      --weight-preflight convnext_tiny=/workspace/artifacts/preflight/convnext_tiny_weights.json \
      --weight-preflight vit_small_patch16_224=/workspace/artifacts/preflight/vit_small_patch16_224_weights.json \
      --weight-preflight swin_tiny_patch4_window7_224=/workspace/artifacts/preflight/swin_tiny_patch4_window7_224_weights.json \
      --weight-preflight dinov2_vits14=/workspace/artifacts/preflight/dinov2_vits14_weights.json \
      --output /workspace/artifacts/preflight/hardware.json

Completion criterion: every summary identifies CUDA, records positive reserved memory below visible VRAM, preserves effective batch 64, authenticates data/config/weights, and produces a measured 15-run training-plus-validation projection acceptable for the rental budget.

## 7. Full training and resume

One run:

    export WANDB_MODE=online
    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model resnet50 --seed 17 --run-dir /workspace/artifacts/runs/resnet50-seed17 --preflight /workspace/artifacts/preflight/hardware.json --max-device-batch "$BATCH"

Resume with exactly the same config, paths, topology, and batch:

    python -m sac_qutab train --config configs/core.yaml "${COMMON[@]}" --model resnet50 --seed 17 --run-dir /workspace/artifacts/runs/resnet50-seed17 --resume /workspace/artifacts/runs/resnet50-seed17/checkpoints/latest.pt --preflight /workspace/artifacts/preflight/hardware.json --max-device-batch "$BATCH"

Execute the complete 5×3 frozen experiment only after review-freeze:

    python -m sac_qutab run-all --config configs/core.yaml "${COMMON[@]}" \
      --pair-manifest validation=/workspace/artifacts/counterfactuals/validation.jsonl \
      --pair-manifest calibration=/workspace/artifacts/counterfactuals/calibration.jsonl \
      --pair-manifest final_test=/workspace/artifacts/counterfactuals/final_test.jsonl \
      --review-freeze /workspace/artifacts/review/review_freeze.json \
      --preflight /workspace/artifacts/preflight/hardware.json \
      --max-device-batch "$BATCH"

run-all trains 15 runs, performs validation and calibration, freezes all 15 checkpoints/detectors/control distributions before final-test inference, and generates protocol-v4, statistics-v3, and report artifacts. Statistics contain five within-model analyses and all 10 unique right-minus-left cross-model contrasts with clustered intervals and Holm correction.

## 8. Export before deleting the instance

Create an archive and checksum while the instance still exists:

    cd /workspace
    tar -czf sac-qutab-export.tgz \
      artifacts/preflight artifacts/runs artifacts/freezes artifacts/detectors \
      artifacts/evaluations artifacts/statistics artifacts/report configs/core.yaml \
      sac-qutab/requirements.txt sac-qutab/pyproject.toml
    sha256sum sac-qutab-export.tgz > sac-qutab-export.tgz.sha256
    rsync -av --progress -e 'ssh -p <LOCAL_SSH_PORT>' sac-qutab-export.tgz* '<LOCAL_USER>@<LOCAL_HOST>:<DESTINATION>/'

At the destination:

    sha256sum -c sac-qutab-export.tgz.sha256
    tar -tzf sac-qutab-export.tgz

Completion criterion: checkpoints, local logs, W&B identity, resolved configs, preflight evidence, frozen protocol, evaluations, statistics, reports, and checksum are readable outside Vast.ai. Only then terminate/delete the rental.

## 9. Evidence boundary

Locally verified: configuration parsing, 15-run generation, global-batch preservation, strict adapter behavior through deterministic no-network doubles, DINOv2 positional-adaptation allowlist, checkpoint/resume, W&B lifecycle, governance, protocol, 10-pair statistics, reporting, and synthetic CPU wiring.

Still instance-dependent: installation of pinned timm against the selected PyTorch build, canonical weight structural validation, real EuroSAT I/O, actual CUDA forward/backward for all five models, common-batch memory fit, throughput, worker selection, interruption recovery, online W&B visibility, 15-run duration, convergence, and every scientific result.
