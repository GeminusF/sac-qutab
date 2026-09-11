# Data and pretrained weights

The executed study uses EuroSAT RGB, 27,000 images in ten classes. Obtain the dataset through the authors' project at https://github.com/phelber/EuroSAT and follow its current distribution and licensing terms. Keep images outside Git in `data/eurosat/`, retaining class-relative names. The frozen manifest and split hashes are in the full evidence record; a newly generated split is not interchangeable with the reported split.

Models are `resnet50.tv_in1k` and `vit_small_patch16_224.augreg_in1k`. Obtain the exact checkpoints identified in the frozen protocol from their upstream timm model cards, then use the existing `validate-weights` CLI before training. The trainer never downloads weights.

Do not silently substitute timm defaults or regenerate final-test results to resolve a checksum mismatch. A mismatch means the resource differs from the frozen study.

The headline replay needs only saved numeric outputs and fitted detector coefficients, not raw EuroSAT images or pretrained files. Redistribution of raw images, pretrained weights and full checkpoints remains subject to upstream permissions. The headline release excludes those binaries.

For complete hardware preflight and prepare/finalize commands see [A100 runbook](A100_TRAINING_RUNBOOK.md). Historic paths in that record describe the original execution; provide your own local data and private administrative paths.
