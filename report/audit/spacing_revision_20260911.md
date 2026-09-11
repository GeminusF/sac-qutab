# Paragraph-linked spacing revision

## Build and scope

The previous 27-page build (18 main, nine appendix) used forced new pages for wide visuals. The current Tectonic build has 19 pages (13 main, six appendix). Main-body font size, column gap and margins are preserved. A one-column IEEEtran outer canvas holds balanced two-column text bands and inline full-width visuals. No cuted/strip output routine is used.

The approved Figure 1 PDF is unchanged. Numeric CSVs, original supplementary assets, original attention arrays and frozen evaluations are unchanged. Compact plots are presentation copies from saved results; gallery images use authenticated original bytes and attention uses existing maps. No model forward pass, training or detector fitting was performed during this spacing revision.

## Placement map and page review

| Visual | Page | Placement review |
|---|---:|---|
| Figure 1 | 3 | Immediately after Method introduction on page 2; no intervening text |
| Table I | 5 | After dataset introduction, within its column |
| Table II | 6 | After models introduction at the end of page 5 |
| Table III | 7 | After clean-performance introduction; immediate interpretation occupies adjacent column |
| Figure 2 | 7 | After RQ1 resolution-response paragraph |
| Table IV | 8 | After severe-minus-mild interpretation, which continues from page 7 |
| Figure 3 | 8 | After representation-response paragraph |
| Figure 4 | 9 | After scatter population and interpretation paragraph |
| Table V | 9 | After absolute detector scores and mean gains |
| Figure 5 | 10 | After dedicated per-seed uncertainty paragraph at the end of page 9 |
| Figure 6 | 11 | After class-wise paragraph at the end of page 10 |
| Table VI | 12 | After planned-versus-executed introduction |
| Class counts and per-seed tables | 14 | After their appendix introductions |
| Figure 7 | 14 | After learning-curve introduction |
| Full-condition table | 15 | After condition-summary introduction |
| Figures 8 and 9 | 16 | Each follows its selection/continuation text |
| Figure 10 | 17 | After rollout-method and repair explanation |
| Population and interval tables | 17, 18 | After their respective explanatory paragraphs |

All final pages 1–19 were individually inspected as Poppler-rendered 1400-pixel page images. A subsequent repeat build was rendered at the same resolution and all 19 pages were pixel-identical to the reviewed pages. No clipped content, overlapping objects, detached captions, blank pages or unrelated text between a visual and its introduction was observed. Modest bottom space on pages 2 and 10 is caused by the next indivisible wide figure; pages 13 and 19 have natural final-section remainder. Large artificial per-visual empty pages were removed.

Default Poppler text extraction interleaved the page 8 columns. Visual review and `pdftotext -raw` confirm both initially flagged paragraphs are complete. The automated main-prose prefix check now follows PDF content order; it is a smoke test, not a proof of every character's integrity. Numeric sources and headline claims are separately checked by the package validator.

## Remaining acceptance limits

- The main report occupies 13 pages, one above the 10–12 target. Scientific content and body type were not reduced to force that target.
- Figure 1 has approximately 7 pt native text, not the previously assumed 10 pt. Its preserved .82 text-height placement reduces that text to approximately 5.3 pt. The unchanged portrait diagram therefore does not satisfy an 8 pt floor. Its geometry must be redesigned to resolve this; no unapproved redesign was made.
- Final author approval, anonymous repository access, course team authorization and release-tag verification remain owner checks.

## Reproduction

Run `compact_response_figures.py` for presentation assets, then `render_manuscript.py`. Compile with Tectonic, rerun `audit_layout.py`, render and inspect every page, and update `pdf_qa.json` only after review. Run `validate_package.py` last. The generator was rerun without a section diff. PDF hash and page labels are recorded in `layout_measurements.json` and `pdf_qa.json`; the validator authenticates those records rather than assuming a fixed page count.
