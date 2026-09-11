# Introduction {#sec:introduction}

A land-cover classifier can score well on clean image patches and still respond unpredictably to changes in their appearance or resolution. Confidence may remain high after the predicted class changes. Internal features may move while the class label stays fixed. An evaluation based only on clean accuracy misses these differences.

EuroSAT provides a land-use/land-cover benchmark derived from Sentinel-2 imagery [@eurosat]. Test accuracy on its RGB patches measures agreement with dataset labels on a particular partition. Evaluating sensitivity also requires comparing the model's responses to the same image before and after a controlled change. Holding image identity fixed separates differences between source images from the response to the transformation.

Semantic Analysis of Counterfactuals: Quality Under Transformations and Adaptive Benchmarking (SAC-QUTAB) studies RGB gain, Fourier amplitude mixing, and resolution reduction. Each source keeps the same randomization or donor choice across three severity levels. Identity shams check the transformation processing paths, while different-class controls provide a reference for representation displacement. The executed comparison uses ResNet-50 and ViT-Small/16, each with training seeds 17, 29, and 43. Four data partitions separate training, validation-based selection and normalization, detector fitting, and final evaluation.

Three questions guide the study:

- **RQ1:** How do predictions change with transformation family and severity?
- **RQ2:** What different behaviors are revealed by prediction stability and representation stability?
- **RQ3:** Does representation information improve failure detection when added to the specified output/confidence features?

The study provides a reproducible paired evaluation of the two trained systems. Matching sources across severity levels makes their dose responses directly comparable. Raw feature distances and their control-based percentiles describe representation change on different scales. A nested detector comparison then tests whether representation information improves failure detection on held-out data. All three seeds contribute to each model's results, including runs with lower clean performance.

The comparison reveals a trade-off: ResNet-50 has higher mean clean accuracy, while ViT-Small/16 retains more accuracy under resolution reduction. Representation instability improves the specified output-only detector in all six runs. The evidence comes from two pretrained variants and one dataset split that groups similar content. No human semantic-preservation review was conducted, and severe transformations can remove information needed for the original label. Results are therefore interpreted as sensitivity to the chosen transformations, with correctness measured against the source labels.

The method and experimental setup define the paired measurements and the information available at each stage. Results address RQ1–RQ3, followed by a discussion of alternative explanations and limits on generalization. The appendix supplies per-seed details and selected visual examples.
