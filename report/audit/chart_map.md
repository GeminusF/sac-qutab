# Figure and table contracts

## Document mapping

The explicit user format is IEEE LaTeX, not an HTML analytics report. Its technical-report functions map as follows: answer-first summary → Abstract; scope/definitions/method → Method and Experimental Setup; evidence → Results tables/figures; interpretation/alternatives → Discussion; limitations and source boundaries → Discussion, Reproducibility and appendix. No executive-dashboard or business-KPI framing is imposed.

## Rebuilt main figures

| File / destination | Question and source | Encodings / aggregation | Uncertainty and safeguards |
|---|---|---|---|
| `framework` / Method | How are fitting and test roles separated? `source_audit.json.split`, frozen protocol | Exact partition counts; arrows indicate workflow, not statistical independence | No invented samples; validation shared for checkpoints and ECDF; no-human-review note |
| `prediction_response` / RQ1 | Does family/severity change prediction? `conditions.json` | Six panels: accuracy (%) and JS (nats), three fixed doses, three-seed means | Sample SD, shared y-scales within metric; lines only join measured doses |
| `representation_response` / RQ2 | What do raw and normalized representations show? `conditions.json` | Raw cosine vs control-ECDF percentile, model/seed-specific controls | Sample SD; bottom panels explicitly have different scales; low ECDF is not invariant prediction |
| `detector_deltas` / RQ3 | Does adding representation improve discrimination? `primary_statistics.json.primary_per_seed` | Point and interval for all six runs, AP and AUROC panels; zero reference | Stored paired source-group 95% bootstrap CI, conditional on fitted models; not seed SD or simultaneous coverage |
| `confidence_representation_scatter` / RQ2 | How do confidence drop and normalized instability coexist? Six authenticated `pair_metrics.jsonl` files | Two common-scale model panels; all 109,350 non-sham observations per panel; transparency, no subsampling | Positive x means confidence drop; y is a control percentile, not failure probability. Includes initially incorrect predictions; repeated sources and pooled conditions are not independent points |
| `class_heatmap` / controls | Which classes show severe prediction changes? `classes.json`, original class CSV | Ten classes × three families per model; three-seed mean consistency; printed 0–1 values | All class sources, not RQ3 clean-correct subset; shared viridis 0–1 scale; descriptive only. RGB gain and full-spectrum amplitude mixing named explicitly |
| `learning_curves` / appendix | How do all selected runs differ during training? original `events.jsonl` validation events | One-based epoch; macro-F1; all three seed curves | Seed distinguished by line style; focused 0.65–1 y-range disclosed; no training-time comparison |

ResNet uses blue (#2864A0), circles and solid lines; ViT uses orange (#C27A23), squares and dashed lines where both coexist. Learning plots use within-model line styles for seeds. Heatmaps use one sequential scale, since color there denotes value, not architecture. Main plots are vector PDFs with PNG inspection copies; figure typography is separate from the unchanged IEEE body font.

Ten report PNGs were visually inspected across the original and revision passes. Figure 1 uses the approved vector pipeline PDF with SHA-256 `fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71`; its labels fit within the boxes and remain readable in the assembled report. Heatmap columns are grouped beneath model headings without overlap. The revised quantitative plots use a 7.16-inch wide layout close to the intended full-width print size. Gallery panels retain original pixels with readable metadata. Assembled-PDF review is recorded separately in `pdf_qa.json`.

## Qualitative appendix and logos

- `qualitative_failure_gallery_1` and `_2`: two clean/transformed pairs each, preserving the original deterministic top-four ordering. Eligible candidates are mild RGB-gain or amplitude-mixing failures with clean confidence at least 0.99; all four selected cases are RGB gain. Original JPEG and transformed float32-cache SHA-256 values match the manifest/evaluations. Confidence is printed to six decimals and remains a rounded prediction score. These are selected examples, not prevalence estimates or semantic validation.
- `attention_rollout_repaired`: the original overlay combines 64-by-64 image coordinates and a 224-by-224 map without a shared extent. The repaired 2-by-2 panel uses the exact gallery example 2 and frozen ViT seed17 checkpoint. A single CPU forward call processed two inputs. Predictions match the archive, maximum confidence difference is 0.000153, and both display layers use the same 224-by-224 extent. Each map is separately normalized; the figure is descriptive, not causal attribution. The defective original remains archived.
- Logos: AZCON → NAIC → Academy → SAC-QUTAB above the title. Aspect ratios and retained pixel colors are unchanged. Crop records and original/derived hashes appear in `visual_revision.json`. Academy's bounds use alpha ≥16 plus four pixels of margin to exclude almost-transparent stray pixels.

## Table mapping

- `split`: four realized image/group counts.
- `models`: actual variants, representation dimensions, training learning rates and devices.
- `clean`: six clean metrics, model means and sample SD.
- `contrasts`: paired severe-minus-mild effects for three metrics, three families and both models.
- `detectors`: all five frozen score variants, plus paired AP/AUROC increments.
- `deviations`: planned versus executed design; no planned extension treated as completed.
- Appendix: `class_counts`, `per_seed_clean`, `full_conditions`, `population`, `delta_ci`.

Nine numeric table fragments come from `build_assets.py`; the two design tables live in `render_manuscript.py`. CSV copies of numeric tables preserve the displayed strings; audit JSON is the full-precision analysis source.

## Archived supporting material

The 15 original CSVs and 47 original PNGs are copied byte-for-byte into `supplementary/` by `build_assets.py`. They include additional reliability/confusion plots, CKA, throughput, qualitative and attention artifacts. Copy integrity is checked by `validate_package.py`. Copy equality is not independent scientific or visual validation of all original plots. Headline tables/figures are rebuilt from audited sources instead. Original qualitative/attention metadata is retained, and no recomputed or generated image replaces unavailable experimental pixels.
