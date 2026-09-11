# Discussion and Limitations {#sec:discussion}

## Model rankings depend on the tested change

The model ranking depends on the change applied to the image. ResNet's clean-accuracy advantage coexists with a larger loss under resolution reduction, while ViT's resolution advantage does not extend to RGB gains. Averaging across families obscures these differences because resolution dominates the aggregate contrast in normalized instability. When the expected image changes are known, the family-specific responses provide the more relevant comparison.

Several factors could contribute to the observed ordering. The models differ in pretraining recipe, representation dimensionality, fine-tuning learning rate, software, and training hardware. Early stopping also selects widely separated ViT epochs. These factors were not varied independently, so the resolution result cannot be attributed solely to global attention. Isolating an architectural explanation would require controlling the training recipe and environment.

## Designed transformations and scene meaning

The absence of human review limits the interpretation of prediction changes. Full-spectrum amplitude mixing can alter spatial structure as well as appearance, RGB clipping can change local cues, and downsampling can erase small objects or boundaries. Same-class donors and retained source phase constrain the operation, but a reviewer might still assign a different label afterward. Reported failures are measured relative to the source dataset labels.

Information loss is an alternative explanation to shortcut reliance, especially under severe resolution reduction. The mild RGB-gain examples in Appendix \ref{app:qualitative} show that even high clean confidence can precede a changed, incorrect prediction. Their deliberate selection leaves both frequency and causal explanation unresolved. Attention rollout describes the model's attention pattern but cannot determine whether the original label remains appropriate.

## What the detector gains establish

The positive within-run gains support adding normalized feature displacement to transformed uncertainty and entropy for the specified task. Richer output features, including clean/transformed JS and probability-vector differences, remain untested and might recover some of the same information. Their evaluation would need to be identified as a separate analysis, with the original frozen baseline retained as the reference.

The need for a clean reference image restricts the evaluated detector to paired benchmark diagnosis. Monitoring incoming satellite images would require a different evaluation design. The present task also excludes existing clean errors and pools conditions with different failure rates. AP and AUROC assess ranking within that population; they do not supply a deployment false-alarm rate, calibrated failure probability, or operating threshold.

## Sampling and uncertainty

Content-based grouping addresses the exact and near-duplicate relationships defined by the implementation but does not enforce geographical separation. The study uses one RGB benchmark without an independent domain, season, or multispectral evaluation. Three training seeds provide a limited sample of optimization variability. Additional bootstrap resamples cannot supply evidence from more training runs or test domains.

Seed SD describes dispersion across trained runs, while per-seed source-group bootstrap CIs describe finite-source variation conditional on a checkpoint and fitted detector. The aggregate two-level interval also resamples the three available seeds, providing limited evidence about less common training outcomes. Multiple-testing corrections apply only to the families defined in the setup. Class heatmaps and post-hoc correlations consequently retain a descriptive interpretation.

## Planned versus executed study

Table \ref{tab:deviations} records the main changes from the proposal and older template. Model variants, split roles, transformation definitions, and results follow the executed study. Optional spatial, masked-autoencoder, multispectral, and gamma extensions were not evaluated. The reporting repairs and visualization issue are documented separately in Appendix \ref{app:provenance}.

{{deviations}}

Human review of label preservation on a prespecified sample would clarify the meaning of the observed failures. Additional seeds and externally held-out domains would address training and sampling uncertainty. Richer output baselines would test how much of the paired signal can be recovered without representation access. Each follow-up requires a new evaluation design that preserves the current final test as a completed experiment.
