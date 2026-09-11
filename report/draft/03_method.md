# Method {#sec:method}

SAC-QUTAB compares each image with a deterministically transformed version of itself. Keeping the source fixed allows prediction and representation changes to be measured on matched inputs. Label consistency records whether the predicted class changes, probability-based metrics measure changes in the output distribution, and representation distance measures movement in feature space. Figure \ref{fig:framework} shows how the four data partitions support these measurements. The term "counterfactual" refers to an image pair constructed for this benchmark; it does not imply a real-world causal intervention or verified preservation of scene semantics.

{{framework}}

## Paired observations and failure definition

Let \(x_i\) be an RGB image with dataset label \(y_i\). For transformation family \(f\), severity \(s\), and deterministic latent choice \(\xi_{if}\), construct \(x'_{ifs}=T_{fs}(x_i;\xi_{if})\). Reusing the latent choice across severities changes the dose while keeping the channel-gain direction or donor fixed. A trained model produces class probabilities \(p_i=p_\theta(x_i)\), \(q_{ifs}=p_\theta(x'_{ifs})\), and representations \(z_i,z'_{ifs}\). The predicted label is the probability argmax. All models and seeds use the same pair manifest, so their responses can be compared on matched inputs. The manifest records source identity, grouping, label, transformation parameters, donor identity where applicable, and cache digest.

Prediction consistency is \(L_{ifs}=\mathbb{1}[\arg\max p_i=\arg\max q_{ifs}]\). A classifier can be consistently wrong, so transformed accuracy separately compares \(\arg\max q_{ifs}\) with \(y_i\). A counterfactual failure requires a correct clean prediction followed by an incorrect transformed prediction:

\[
F_{ifs}=\mathbb{1}[\arg\max p_i=y_i]\,
         \mathbb{1}[\arg\max q_{ifs}\ne y_i].
\]

For RQ3, only clean-correct sources and the nine non-sham family--severity combinations are eligible. Conditioning on a correct starting prediction separates correct-to-incorrect transitions from errors already present on the clean image. The resulting population differs by model and seed. Within each population, the target is whether the transformed prediction is incorrect. Measuring representation change requires a clean reference image, so the evaluated detector operates as an offline diagnostic on paired inputs.

## Transformations and their limits

All interventions act on the native 64-by-64 RGB tensor, with pixel values initially in [0,1]. The result is then resized to 224-by-224 using bicubic interpolation with antialiasing and normalized with ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). Applying transformations before this common preprocessing keeps them defined at the dataset's native resolution. Upsampling matches the model input size without adding independent spatial measurements.

**RGB gain (spectral appearance).** For each source, draw three deterministic values \(u_c\) uniformly from [-1,1] and retain this direction across severity. The implementation is

\[
\begin{gathered}
T_{\mathrm{gain},s}(x)_c=\operatorname{clip}((1+r_su_c)x_c,0,1),\\
r_s\in\{0.05,0.10,0.20\}.
\end{gathered}
\]

The radius scales a fixed channel-gain vector; the vector is not normalized to unit Euclidean length. RGB gain probes sensitivity to channel appearance in the available three-band input. The operator has no calibrated atmospheric or sensor-response interpretation and does not use the additional Sentinel-2 bands. Clipping can contribute to the response, particularly for bright pixels.

**Fourier amplitude mixing (texture).** A same-label donor is chosen deterministically from the training partition. Source and donor cannot share identity, exact content hash, duplicate group, or a difference-hash distance at or below the configured threshold. For channel-wise Fourier amplitude \(A\) and phase \(\phi\),

\[
T_{\mathrm{amp},s}(x)=\operatorname{clip}\!\left(
 \Re\mathcal{F}^{-1}\!\left[
 ((1-\lambda_s)A_x+\lambda_s A_d)e^{j\phi_x}
 \right],0,1\right),
\]

where \(\lambda_s\in\{0.10,0.25,0.40\}\), and the donor is fixed across severity. Mixing covers the full amplitude spectrum. Unlike the low-frequency replacement used in FDA [@fda], the operator blends every frequency and performs no target-domain adaptation. Retaining source phase and choosing a same-class donor constrain the transformation, but neither establishes that the transformed image preserves its label. Since donor selection uses the source label, reproducing this benchmark also requires access to that label.

**Resolution reduction.** The native tensor is downsampled to 48, 32, or 16 pixels per side, restored to 64-by-64, and then passed through the common 224-by-224 preprocessing. Both intervention resizes use bicubic interpolation, antialiasing, and non-aligned corners; the restored result is clipped to [0,1]. These settings intentionally discard progressively more spatial detail. Severe degradation can remove genuine class evidence, so its failures cannot all be attributed to shortcut reliance.

Severity orders doses within each family. The three families are not matched for perceptual or semantic damage. A larger response to resolution reduction therefore describes sensitivity on the chosen intervention grid; it cannot rank responses to equally difficult physical disturbances.

## Sham, negative control, and condition-selection policy

Each family includes a sham: zero gain radius, zero amplitude-mixing strength, or native-size resolution processing. Passing an identity case through the same code checks whether the processing path itself changes predictions. Shams are excluded from the detector task. A different-class negative-control image is separately selected from the source's own partition using the content/group exclusion rules. Its representation supplies a reference for dissimilarity; the control image is excluded from the set of transformed-source failure examples.

No human semantic-preservation review was performed. The frozen policy retained every predeclared non-sham condition. Legacy fields such as `audit_qualified` record inclusion under that policy. They do not record human judgement, and the synthetic review fixture served only as a software check. Quantitative results and visual examples are therefore evaluated against the original dataset labels.

## Prediction and representation metrics

The predictive-distribution response is Jensen--Shannon (JS) divergence using natural logarithms:

\[
J(p,q)=\tfrac12 D_{\mathrm{KL}}(p\Vert m)
       +\tfrac12 D_{\mathrm{KL}}(q\Vert m),
\qquad m=(p+q)/2.
\]

JS lies in [0,\(\log 2\)] nats and detects probability redistribution even when the predicted class stays the same. Complementary measures are transformed entropy, the signed confidence drop \(\max p-\max q\), and its absolute value. A negative signed drop means that confidence increased after transformation; it does not by itself indicate improved correctness.

Raw representation similarity is cosine similarity \(C(z,z')=z^\top z'/(\lVert z\rVert\lVert z'\rVert)\). Raw dissimilarity is \(D=1-C\). ResNet-50 uses the 2,048-dimensional pre-logit global-pooled vector, while ViT-Small/16 uses the 384-dimensional final normalized classification token. Their coordinates and dimensionalities differ, so raw cosine is interpreted primarily within a model.

A different-class reference distribution places each displacement on a model-specific scale. For each model and seed, validation controls define an empirical cumulative distribution function (ECDF),

\[
U(D)=\frac{1}{N_v}\sum_{i=1}^{N_v}
 \mathbb{1}[D(z_i,z_i^{\mathrm{control}})\le D],
\qquad N_v=4050.
\]

Fitting the mapping on validation data gives calibration and final test a reference fixed before their evaluation. A value near one means that the transformed pair is more dissimilar than most validation different-class pairs; zero means its dissimilarity lies below the observed control distribution. The percentile measures relative displacement, not failure probability. Each model retains its own representation geometry even after normalization. Values near the ECDF floor can conceal small within-model changes.

Linear centered kernel alignment (CKA) is an auxiliary statistic for comparing sets of representations [@cka]. For centered feature matrices \(X,Y\), it is \(\lVert X^\top Y\rVert_F^2/(\lVert X^\top X\rVert_F\lVert Y^\top Y\rVert_F)\). Cosine and normalized instability measure individual pairs, whereas CKA measures relationships across collections. None directly measures semantic understanding.

## Nested failure detectors

RQ3 tests the additional information provided by representation displacement. The output-only detector receives transformed uncertainty \(1-\max q\) and entropy \(H(q)\); the augmented detector receives the same features plus \(U(D)\). Both use the same fitting procedure and test population, so the comparison measures the gain from adding that feature. The baseline is limited to two output summaries. Logits, the full probability vector, and clean/transformed JS are outside this comparison.

Both detectors standardize their input features using calibration-partition means and standard deviations, then fit L2-regularized logistic regression with \(C=1\), the LBFGS solver, tolerance \(10^{-4}\), at most 1,000 iterations, random state zero, and no class weighting. They are fitted separately per model and seed. Neither standardization nor classifier fitting uses final-test examples. The protocol requires at least 25 positive calibration and test examples and both target classes; all six reported final-test populations meet that criterion.

Three unlearned scores provide context: transformed uncertainty, transformed entropy, and normalized representation instability alone. They are evaluated on exactly the same eligible rows as the two learned detectors. Discrimination is summarized by AUROC and AP. AP is the sum of precision values weighted by recall increments, as implemented by scikit-learn [@ap]. This definition differs from trapezoidal integration of the precision--recall curve. The AP baseline depends on positive prevalence. The main detector contrast is computed within each population as the augmented detector's AP or AUROC minus the corresponding value for the output-only detector.
