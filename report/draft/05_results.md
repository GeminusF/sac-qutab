# Results {#sec:results}

## Clean performance and seed variability

ResNet-50 achieves higher clean accuracy with less variation across seeds under the executed recipes. Mean final-test accuracy is 98.63% for ResNet-50 and 96.77% for ViT-Small/16, with sample SDs of 0.04 and 1.21 percentage points, respectively. Macro-F1 follows the same ordering (Table \ref{tab:clean}). These clean scores provide the reference for measuring transformation effects.

{{clean}}

ResNet's mean expected calibration error (ECE) is 0.0080, versus 0.0116 for ViT, using 15 equal-width confidence bins. Brier score and negative log-likelihood also favor ResNet. Calibration and accuracy nevertheless measure different properties: the lowest-accuracy ViT seed has ECE 0.0067, lower than that of the highest-accuracy ViT seed. A small aggregate calibration gap can coexist with lower accuracy. ECE also depends on binning and the evaluated population [@calibration].

ViT seed 17 attains 98.15% clean accuracy, while seeds 29 and 43 attain 96.25% and 95.90%. All three contribute to the mean. The selected validation epochs and learning curves document this variation across runs, although they do not identify its cause.

## RQ1: sensitivity depends strongly on the transformation family

Resolution reduction produces the largest deterioration on the tested dose grid. ResNet accuracy falls from 87.89% at mild reduction to 49.47% at moderate and 12.94% at severe reduction. ViT falls from 92.89% to 77.17% and 43.00%, respectively. ViT therefore retains higher accuracy at all three tested resolution levels despite its lower clean accuracy. Figure \ref{fig:prediction} shows accuracy alongside JS divergence, separating label correctness from changes in the probability distribution.

{{prediction}}

ResNet retains the advantage under the other two families. Under severe RGB gains, accuracy is 92.14% for ResNet and 85.85% for ViT; under severe amplitude mixing, the means are 94.41% and 93.59%. Model performance must therefore be described by transformation family. The intervention scales are not perceptually matched, and severe resolution reduction may remove information needed to identify the original class.

Table \ref{tab:contrasts} reports severe-minus-mild effects for matched source images. ResNet's mean resolution JS increase is 0.5219 nats, a large change within the possible JS range of 0 to \(\log 2\). Its RGB-gain and amplitude-mixing increases are much smaller. Label inconsistency, distributional change, and normalized feature change have different units, so their numerical magnitudes should be interpreted within each metric.

{{contrasts}}

The lowest dose can also produce a small accuracy increase. Mean mild amplitude-mixing accuracy is 98.73% for ResNet and 97.13% for ViT, slightly above their respective clean means. Accuracy can rise when corrections of initially wrong predictions outnumber correct-to-incorrect transitions. Label consistency records both kinds of changes. The observed increases are descriptive outcomes on this held-out partition; they do not establish that amplitude mixing would improve generalization if used during training.

## RQ2: normalized feature stability and prediction stability are not interchangeable

Resolution reduction also produces the largest representation changes within each model (Figure \ref{fig:representation}). At severe reduction, ResNet's mean raw cosine similarity is 0.3929 and its ECDF instability is 0.6202; ViT's corresponding values are 0.5550 and 0.0876. The lower ViT ECDF value places its displacement at a lower percentile of its own different-class control distribution. Because the reference distributions are model-specific, the percentiles do not equate absolute distances in the 384- and 2,048-dimensional spaces.

{{representation}}

Under resolution reduction, the paired increase in normalized instability is 0.6131 ± 0.0361 for ResNet and 0.0874 ± 0.0069 for ViT (mean ± sample SD). Averaging the ViT-minus-ResNet difference-in-differences over all three families gives -0.1744, with a stored two-level bootstrap 95% CI of [-0.1851, -0.1633]. The group-wise aggregate Wilcoxon result is reported as \(p<0.001\); its Holm family contains only the single active model pair. Resolution dominates this aggregate, so the average does not describe an equal effect across RGB gain, amplitude mixing, and resolution reduction.

Severe RGB gains illustrate why normalized feature stability and prediction stability can diverge. Mean ECDF instability is only 0.00285 for ResNet and 0.00537 for ViT, yet label consistency falls to 92.61% and 86.77%. Features can remain close relative to different-class controls while crossing a decision boundary. The raw-cosine panels in Figure \ref{fig:representation} retain within-model changes that are compressed near the ECDF floor.

Figure \ref{fig:scatter} relates confidence drop to normalized representation instability for individual pairs. Each panel contains 109,350 non-sham observations, pooling three seeds and nine conditions for 4,050 sources. In both models, a dense band near the ECDF floor spans a range of confidence changes. A low control percentile therefore does not determine the size of the confidence change. Sources appear repeatedly across seeds and conditions, so the points are dependent observations; the display does not support a causal interpretation.

{{scatter}}

The scatter includes initially incorrect predictions. RQ3 restricts evaluation to initially correct sources and tests whether the representation feature improves detection of their subsequent failures.

## RQ3: representation adds discrimination beyond the specified output baseline

Adding normalized representation instability improves both AP and AUROC in every model/seed comparison. Mean paired AP gains are 0.432 ± 0.113 for ResNet and 0.402 ± 0.032 for ViT. Mean paired AUROC gains are 0.166 ± 0.087 and 0.114 ± 0.023. The ± values are sample SDs across three seeds. Table \ref{tab:detectors} gives all five score variants and the paired gains.

{{detectors}}

Figure \ref{fig:detectors} shows the within-seed source-group bootstrap intervals for the paired gains. Every displayed interval lies above zero, although the twelve intervals do not provide simultaneous coverage.

{{detector_figure}}

The eligible population comprises 35,937--35,964 pairs per ResNet seed and 34,956--35,775 per ViT seed. Positive prevalence is 18.05--18.55% for ResNet and 10.30--13.12% for ViT. Both population size and prevalence depend on which sources each run classifies correctly before transformation and which then fail. Since AP depends on prevalence, the absolute scores do not provide a population-matched ranking of the models. Comparing the two detectors within each run holds the population fixed. Appendix \ref{app:detector} reports the exact counts and prevalence.

The representation-only score achieves mean AP of approximately 0.918 for ResNet and 0.878 for ViT. Adding the two output features raises the respective means to approximately 0.939 and 0.898. Much of the gain over the output-only baseline is therefore present in the representation score alone. The evidence supports this feature's value within the tested logistic model, without establishing a need for a more complex fusion architecture. Richer output-only baselines remain an open comparison.

The ResNet output baseline varies substantially across seeds: its AUROC is 0.765, 0.916, and 0.772 for seeds 17, 29, and 43. The augmented detector remains at approximately 0.982--0.985 across the same runs. The resulting gain is consequently smaller for seed 29, as reflected in the per-seed intervals and the SD of the paired gains.

The pooled detector task gives each source one pair in each of the nine conditions. Differences in condition difficulty, especially the high failure rate under resolution reduction, can contribute to pooled discrimination. Equal performance within every family, class, or severity cannot be inferred from that score. The saved condition-prevalence tables document the mixture, but the pooled summary does not supply condition-specific AUROC estimates. Detector calibration, operating thresholds, and performance under deployment shifts were also not established by this evaluation.

## Controls and class-wise behavior

Sham conditions retain label consistency of one across all six runs, confirming that the identity processing paths preserve predicted labels. The check does not extend to semantic preservation under nonzero perturbations. Different-class controls supply the ECDF reference scale using known labels and content exclusions; their purpose is normalization rather than an independent adversarial stress test.

Figure \ref{fig:classes} breaks down severe-condition label consistency by the ten original classes. Each cell averages the same three seeds and includes all final-test sources in the class, including clean errors. The denominator therefore differs from RQ3. The resolution columns show substantial variation across classes that is concealed by a single model mean. All classes are shown, and the cell differences are descriptive; no class-wise significance tests were performed.

{{classes}}

Appendix \ref{app:qualitative} illustrates four failures selected for high clean confidence and mild transformation severity. All four involve RGB gains and prediction switches between PermanentCrop and HerbaceousVegetation. Clean confidence exceeds 0.99 in each case, yet the transformed prediction is incorrect relative to the dataset label. The selection illustrates the failure definition without estimating its frequency. Appendix \ref{app:attention} presents the related attention rollout and documents its rendering repair.
