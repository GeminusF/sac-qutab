> Historical report-build record. The publication uses canonical assets in `report/figures/` and `GeminusF/sac-qutab`; old paths below describe the earlier build.

# QA and submission status

## Completed source-level checks

- Frozen protocol, dataset manifest/split, three pair manifests and six checkpoints authenticated by SHA-256.
- Six evaluation input files matched against their evaluation indices and the primary statistics source list.
- All six NPZ pair-ID orders/shapes checked; every pair's cosine and frozen validation-control ECDF value recomputed in batches.
- Clean accuracy, macro-F1, ECE, Brier score and NLL independently recomputed from one saved original probability vector per source.
- All condition means and counts checked; class-wise accuracy, consistency and ECDF values checked against saved rows.
- Detector eligibility and targets matched to clean-correct non-sham rows; five score variants' AP/AUROC independently recomputed; saved logistic coefficients replayed without fitting or neural-network inference.
- Simple paired primary effect estimates checked. All original bootstrap intervals remain authenticated stored statistics; full bootstrap replication was not rerun.
- Historical representation-SHA defect localized to a stale report-writing path; five wrong index entries recorded, originals untouched.
- Ten report figures visually inspected as PNG assets across the original and revision passes. The revised scatter keeps all 109,350 non-sham observations per model; the 60 heatmap cells are checked against the original class CSV. Gallery inputs and selection are authenticated; no neural inference was used to render the gallery.
- The four gallery rows retain original order, labels, predictions and confidences. Eight real source/cache files were recovered from the migration archive, with matching SHA-256 values.
- The first page now uses one institutional bar in the order AZCON, NAIC and AI Academy. The current NAIC and AI Academy wordmarks match the source files in `report_materials/` by SHA-256. A 0.4 pt rule separates that bar from the SAC-QUTAB logo and two-line paper title.
- Corrected attention rollout generated for the exact gallery example 2 with the frozen ViT seed17 checkpoint. One CPU forward call processed two inputs; predictions match, maximum confidence discrepancy is 0.000153, map/input extents align, and the original-resolution PNG passed visual review.
- Prose revised for purpose-first explanation and readable academic English. Static checks reject authored em dashes and selected formulaic filler. Baseline headline calculations remain unchanged.
- Team-provided names and emails appear in the approved contribution order: Milana Karimova, Sharaf Feyzullayev, Farah Feyzullayev, Jeyhuna Sevdiyeva and Nicat Aghayev. Each author block gives AI Academy and the National Intelligence Center of Azerbaijan; no author-reference marks are used.
- The new title and SAC-QUTAB expansion agree in the visible title, PDF metadata, Abstract and Introduction. The Abstract cites the repository as reference [16] without printing its URL inline.
- Performance tables use bold only for the predeclared, direction-aware best metric cells. Stored CSV values remain unchanged.
- Figure 1 now uses the approved vector `pipeline_diagram_vibrant.pdf`; its canonical SHA-256 is `fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71`.
- Paragraph-linked placement is enforced. Ordinary figures and tables use `[H]`; full-width visuals use non-floating bands between balanced two-column text areas. Per-visual forced page breaks are removed. Appendix tables and title logos remain inline.
- Main-text source, bibliography, appendix, analysis scripts, claim ledger, chart map and build instructions supplied.

`source_validation.json` records the latest automated source-package checks and authenticates the separately completed PDF review through `pdf_qa.json`. The final PDF was built with Tectonic 0.17.0, rendered with Poppler and visually checked page by page.

## Completed PDF checks

- `report.pdf` compiled successfully and its SHA-256 is recorded in `pdf_qa.json`.
- The PDF has 19 pages: 13 main pages including references, followed by six one-column appendix pages.
- All 19 final pages were rendered at 1400-pixel page height and individually inspected. No clipping, overlap, detached captions or blank pages remain. Figure 1's unchanged source contains sub-8 pt text, so the minimum figure-font requirement is not satisfied.
- Cross-reference text contains no unresolved `??` markers.
- All fonts reported by `pdffonts` are embedded and subsetted.
- The build was repeated after the academic editorial revision. Only non-blocking font-shape substitution and underfull-box warnings remain.

## Not completed / owner checks

| Item | Status and reason |
|---|---|
| Corrected attention rollout | Completed and included in the appendix. Local `timm==1.0.29` installation was explicitly authorized by the user. Incorrect original overlay remains archived but excluded from interpretation |
| PDF compile | Completed with Tectonic 0.17.0 |
| Main-page count | Measured: 13 pages including references; appendix starts on page 14. The 10–12 target remains unmet |
| Figure 1 font floor | Unresolved: approximately 7 pt in the approved source, approximately 5.3 pt after placement. Source hash preserved; redesign requires a changed approved asset |
| Assembled PDF visual QA | Completed for pages 1–19, with the font-floor exception recorded |
| Exact placement, long hashes/URLs, email block | Visually checked; the provenance hashes were reflowed and the full-condition table was fitted to the page |
| Repeated successful PDF build | Completed after the final layout edits |
| All authors' final approval / contributions | Names and affiliation confirmed; authors must approve the final submission |
| Five-member team permission | Earlier brief required approval; affiliation confirmation did not explicitly settle this requirement |
| `v1.0-final` and anonymous public repo access | The manuscript cites the user-supplied public repository URL. The requested tag was not created or pushed, and anonymous access must still be checked by the owner before submission |
| Full raw-data/pretraining resource release | Not certified by this report audit; no package or cache cleanup performed |
| Human semantic review | Not performed, not a hidden prerequisite that can be claimed complete |

## Scope statement

The deliverable is a **source-verified, compiled layout revision with disclosed page-count and Figure 1 font limitations**, awaiting author and administrative approval. It is not certified submission-ready. All requested visual families are present. The current placement map is in `layout_measurements.json`; the review is in `spacing_revision_20260911.md`. Original evidence remains under `../all_artifacts/`.
