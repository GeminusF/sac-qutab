# Related Work

## Clean classification and model comparison

Residual networks use skip connections to train deep convolutional models [@resnet], while Vision Transformers process image patches as token sequences [@vit]. These designs offer different ways to represent the same image. Their performance also depends on training: augmentation, regularization, data, and compute interact strongly in ViT training [@augreg]. SAC-QUTAB compares pretrained systems with different pretraining recipes and fine-tuning rates. A performance gap between them cannot therefore be attributed solely to convolution or attention.

EuroSAT provides ten land-use/land-cover classes and 27,000 image patches [@eurosat]. Its RGB images allow the use of standard ImageNet-pretrained models. Since the present split differs from those used in other studies, published EuroSAT accuracies are not directly comparable with the scores reported here. The analysis focuses on changes measured within the same held-out partition.

## Transformation sensitivity and feature reliance

Common-corruption benchmarks evaluate performance under image changes separately from clean accuracy [@corruptions]. Shortcut-learning research examines whether benchmark success relies on cues that fail under different conditions [@shortcut]. Controlled transformations can reveal such sensitivity, but the measured response does not identify which cue caused a prediction. Downsampling, for example, can remove genuine class information.

Texture/shape cue-conflict experiments probe feature reliance by constructing images with conflicting cues [@texture]. Fourier Domain Adaptation modifies amplitude information to address domain mismatch [@fda]. SAC-QUTAB uses the separation of Fourier amplitude and phase to construct a different intervention: full-spectrum convex mixing with a same-class training donor. The model remains fixed during evaluation. The resulting measurements describe sensitivity to this mixing operator; they do not constitute target-domain adaptation or a texture/shape cue-conflict test.

## Confidence, representations, and interpretation

Maximum-softmax confidence is an established baseline for detecting misclassified or out-of-distribution examples [@baseline]. Calibration research distinguishes classification accuracy from the reliability of reported probabilities [@calibration]. Confidence must therefore be evaluated against observed errors. RQ3 tests whether paired representation displacement improves error ranking beyond transformed uncertainty and entropy.

Linear centered kernel alignment (CKA) compares collections of hidden representations [@cka]. Cosine dissimilarity describes an individual clean/transformed pair and supplies the representation feature used by the detector. Average precision (AP) and area under the receiver operating characteristic curve (AUROC) summarize ranking quality across thresholds. AP weights precision by the corresponding recall increments [@ap]. A deployment threshold would require a separate choice based on acceptable errors.

Attention rollout combines attention across layers to approximate how token information is mixed [@rollout]. The selected example in Appendix \ref{app:attention} illustrates the computed pattern and documents a correction to the archived visualization. Rollout alone cannot establish which pixels caused a classification or whether a transformation preserved the scene label.
