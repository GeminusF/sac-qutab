> Historical report-build record. The publication uses canonical assets in `report/figures/` and `GeminusF/sac-qutab`; old paths below describe the earlier build.

# Primary-source reference audit

Checked on 2026-09-10 using the primary publication/author-hosted record, official proceedings, or official project documentation. Bibliography keys match `references.tex`. The sources support the specific background or definition stated below, not the new SAC-QUTAB empirical results.

| Key | Primary record checked | Claim supported / boundary |
|---|---|---|
| `eurosat` | [Author paper](https://arxiv.org/abs/1709.00029), [DFKI publication record](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/10161) | EuroSAT benchmark, 27,000 patches / ten classes; the four-way split is our stored experiment, not the original paper's split |
| `resnet` | [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | Residual/skip-connection design; does not support attribution of our resolution outcome solely to architecture |
| `vit` | [Author paper](https://arxiv.org/abs/2010.11929) | Image-patch token Transformer; actual variant authenticated separately by model card and frozen config |
| `augreg` | [Author paper and publication record](https://arxiv.org/abs/2106.10270) | Data, augmentation, regularization and compute in ViT training; TMLR 2022 publication |
| `corruptions` | [Author paper](https://arxiv.org/abs/1903.12261) | Common-corruption robustness distinct from clean accuracy; our operators are not claimed to be the same benchmark |
| `shortcut` | [Author paper](https://arxiv.org/abs/2004.07780) | Shortcut-learning framing; publisher DOI recorded in bibliography, but no claim of measured causal shortcuts here |
| `texture` | [Author paper](https://arxiv.org/abs/1811.12231) | Cue-conflict/texture-shape study; our full-spectrum mixing is not that experiment |
| `fda` | [Official CVF record](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html) | Low-frequency Fourier amplitude adaptation; explicitly contrasted with our full-spectrum convex mixing. The direct HTML retrieval was restricted, so the official indexed record/paper was used |
| `baseline` | [Author paper](https://arxiv.org/abs/1610.02136) | Maximum-softmax baseline for errors/OOD; our label-induced failures are not arbitrary OOD detection |
| `calibration` | [PMLR proceedings](https://proceedings.mlr.press/v70/guo17a.html) | Confidence calibration versus accuracy; no temperature scaling is inferred in our experiment |
| `cka` | [PMLR proceedings](https://proceedings.mlr.press/v97/kornblith19a.html) | Set-level representation similarity / linear CKA; not per-image semantic equivalence |
| `ap` | [Official API definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html) | Non-interpolated recall-increment-weighted AP, distinct from trapezoidal area. This linked documentation is version 1.5; experimental metadata records scikit-learn 1.9.0. It is a definition citation, not environment evidence |
| `vitcard` | [Official timm model card](https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k) | Exact ImageNet-1k variant and AugReg lineage; frozen hashes, not mutable web content, authenticate local checkpoint |
| `rescard` | [Official timm model card](https://huggingface.co/timm/resnet50.tv_in1k) | Exact pretrained ResNet variant; local checkpoint authenticity comes from protocol hashes |
| `rollout` | [ACL Anthology proceedings](https://aclanthology.org/2020.acl-main.385/) | Abnar and Zuidema, ACL 2020, pp. 4190–4197, DOI 10.18653/v1/2020.acl-main.385. Attention propagation motivates the rollout description; it does not validate causal attribution or the defective local overlay |
| `repository` | [SAC-QUTAB repository](https://github.com/GeminusF/dl-final-project) | Reproducibility entry point supplied by the team. Anonymous public accessibility and the requested `v1.0-final` tag remain owner checks before submission |

No invented source or broad “first-ever” claim is used. Related Work paraphrases these sources; no long direct quotations are included. Experiment counts, effect estimates, population definitions and environment versions are supported by the local evidence ledger, not by general literature citations. The Abstract follows the team's instruction to describe the repository as publicly available; anonymous access and the requested final tag still require an owner check before submission.
