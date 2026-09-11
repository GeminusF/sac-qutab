# SAC-QUTAB report source package

English IEEE conference-style report for ResNet-50 and ViT-Small/16, seeds 17, 29 and 43. This package contains the manuscript and its evidence trail, not a new experiment.

**Status:** spacing revision compiled and visually reviewed on all 19 pages: 13 main pages including references and six appendix pages. Full-width visuals are non-floating bands between balanced two-column text areas, without automatic per-visual page breaks. The 10–12 main-page target remains unmet. Approved Figure 1 is unchanged and contains text below the requested 8 pt print floor; that limitation is not marked as passed. See [current layout audit](audit/spacing_revision_20260911.md) and [QA status](audit/qa_status.md).

## Start here

- `report.tex`: main LaTeX file, including the contribution-ordered 3+2 author grid, team-provided emails, and each author's AI Academy / National Intelligence Center of Azerbaijan affiliation.
- `draft/*.md`: editable English main-text sections. Edit these, then regenerate `sections/*.tex`.
- `appendix.tex`: editable supplementary text and extended tables.
- `references.tex`: complete manually maintained bibliography; no BibTeX run is necessary.
- `figures/`: ten quantitative/workflow/gallery/attention figures in PDF and PNG, plus four trimmed logos in `figures/logos/`. Figure 1 is the approved vector `pipeline_diagram_vibrant.pdf`, copied byte-for-byte to `figures/framework.pdf`. Scatter dots, real gallery pixels and attention maps remain raster content inside their PDF containers.
- `tables/`: nine generated tables, each in LaTeX and CSV. Two additional design tables are defined in `scripts/render_manuscript.py`.
- `supplementary/`: all 15 original report CSVs and 47 original PNGs, plus qualitative/attention metadata. These are archived reference assets, not 47 independently revalidated manuscript figures.
- `audit/claim_evidence.md`: claims, exact evidence paths, denominators and interpretation boundaries.
- `audit/source_audit.json`: corrected per-run source hashes and historical-index discrepancy.
- `audit/chart_map.md`: figure definitions and visual QA scope.
- `audit/reference_audit.md`: primary-source bibliography checks.
- `audit/visual_revision.json`: six-run visual source hashes, exact scatter counts, heatmap cells, four gallery selections, verified image/cache hashes and logo trim records.
- `audit/presentation_revision_20260911.md`: approved title, author, logo, repository-reference and metric-emphasis changes, with the final build hash and QA result.
- `audit/attention_repair_status.json`: exact checkpoint/input hashes, one-call/two-input scope, prediction and confidence replay checks, runtime versions, output hashes and visual-review record for the corrected attention figure.

Original research artifacts remain immutable. Full evidence is provided separately from this source package. The main manuscript uses the requested structure but replaces obsolete method/model descriptions with the executed study. Original evidence is not silently repaired.

The revised prose explains the purpose of the paired design, split roles, controls, ECDF scale and nested detector comparison. Authored prose has no em dashes or selected formulaic filler; technical meanings and headline calculations remain unchanged. AI assistance is disclosed. The institutional bar above the title contains AZCON, NAIC and AI Academy in that order. A rule separates it from the SAC-QUTAB logo and expanded title; the marks do not imply institutional endorsement.

## Compile the document

Work from this `report/` directory, so relative figure and section paths resolve.

```powershell
Set-Location report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

If using an existing TeX distribution without `latexmk`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively, upload the contents of this directory to a LaTeX service and select `report.tex` as the main document, with pdfLaTeX. For document compilation, `audit/`, `draft/`, `scripts/`, and `supplementary/` are not required; keep `sections/`, `tables/`, `figures/`, `appendix.tex` and `references.tex` alongside the main file.

Dependencies: IEEEtran, standard T1 fonts/mathptmx, amsmath, amssymb, graphicx, booktabs, array, url, hyperref, placeins, float, capt-of and multicol. The outer canvas is one-column IEEEtran; `multicols` supplies the two-column main body with the original column gap and font sizes. No shell escape is requested. The verified local build used bundled Tectonic 0.17.0; downloading multicol was explicitly authorized. The resulting PDF is authenticated in `audit/pdf_qa.json`.

### Final PDF checks

1. Completed: references and every `\ref` resolve, with no `??` markers in extracted PDF text.
2. Completed: all 19 pages were visually checked for equations, tables, hashes, email block, axes, captions and source placement. Figure 1's sub-8 pt text remains a disclosed exception.
3. Measured: the main report through references is **13 pages**; the appendix begins on page 14. The main-page target remains open.
4. Completed: the build was repeated after the final layout edits and remained clean.
5. Owner action: obtain all authors' final approval, confirm five-member team authorization, verify anonymous repository access and verify the requested release tag before submission.

## Rebuild text only

From the project root, with Python 3:

```powershell
python report/scripts/render_manuscript.py
python report/scripts/audit_layout.py
python report/scripts/validate_package.py
```

The converter implements a deliberately small Markdown subset: headings, bold, inline code, bullets, citations such as `[@eurosat]`, math and named asset tokens. It is not a general Markdown-to-LaTeX system. Generated `sections/` changes are overwritten by this command; keep manual main-text edits in `draft/`.

## Reproduce the quantitative audit and assets

These commands require the unchanged full evidence extraction. Set `SAC_EVIDENCE_ROOT` to its absolute path, or place it at project-root `all_artifacts/`. They do not use the older `artifacts/` tree.

```powershell
python report/scripts/audit_sources.py
if ($LASTEXITCODE -ne 0) { throw 'Source audit failed; do not rebuild from stale audit outputs.' }
python report/scripts/build_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Asset generation failed.' }
python report/scripts/revise_visuals.py --archive /path/to/sac-qutab-full-migration-backup.tar
if ($LASTEXITCODE -ne 0) { throw 'Visual revision failed.' }
python report/scripts/render_manuscript.py
if ($LASTEXITCODE -ne 0) { throw 'Text conversion failed.' }
python report/scripts/validate_package.py
if ($LASTEXITCODE -ne 0) { throw 'Package validation failed.' }
```

Verified local analysis runtime: Python 3.12, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.9.0, Matplotlib 3.11.1. Package versions describe this report build, not a replacement for the frozen training environments. No package installation is performed. Auditing hashes large checkpoint/NPZ files and can take several minutes and substantial memory. The commands above write only within `report/` and never fit a model, run a neural network, or update original evaluations. Saved logistic detector coefficients are replayed as arithmetic checks; no detector is refitted.

Run `revise_visuals.py` **after** `build_assets.py`, since the latter rebuilds the earlier heatmap layout. The revision reads only eight named image/cache entries from the migration TAR and authenticates their bytes. Adjust `--archive` if the backup has moved. Canonical logos are retained in `figures/logos/` and authenticated by `audit/canonical_assets.json`; no external logo folder is required. The eight recovered inputs are retained under `audit/visual_inputs/`; no full dataset extraction was performed.

`build_assets.py` does not regenerate Figure 1. It verifies that `figures/framework.pdf` has SHA-256 `fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71` and stops if the approved asset has been replaced. Single-column visuals use `[H]`; full-width visuals use non-floating bands between two-column text areas. The clean table shares a band with its immediately following interpretation. Appendix tables and title logos remain inline.

Run `python report/scripts/compact_response_figures.py` after rebuilding base assets to reproduce the compact presentation copies. It reads saved results, authenticated gallery pixels and existing attention maps, without training or model inference. Then regenerate the manuscript, compile, rerun `audit_layout.py`, render every page, and refresh the separate manual PDF QA record before `validate_package.py`. A changed PDF is not automatically visually approved.

### Reproduce the corrected attention illustration

`python report/scripts/repair_attention.py --check-only` checks available modules without running a model. The local Python 3.12 runtime now contains the user-authorized `timm==1.0.29` installation. The completed repair used the exact archived clean/transformed pair and the frozen ViT seed17 checkpoint.

Running the optional script without `--check-only` processes exactly two inputs in one CPU forward call. It authenticates the canonical protocol configuration, checkpoint and inputs, checks both predictions and a 0.005 confidence tolerance, then uses a common 224-by-224 display extent. The completed replay's largest confidence difference was 0.000153. Re-running the script returns the status to `GENERATED_AWAITING_VISUAL_REVIEW`; its output must be inspected again before a regenerated manuscript is treated as verified. The incorrect historical overlay remains preserved in `supplementary/`, not presented as a valid spatial explanation.

The bootstrap point estimates and source provenance are checked, but the entire 2,000-replicate bootstrap pipeline is not rerun. Stored intervals remain authenticated results of the original statistics stage. Additional JS–ECDF correlations are explicitly post-hoc descriptive calculations from saved rows, not new primary tests.

## Scientific and release boundaries

- A stale `representations_path` in historical `reporting.py` wrote the last run's SHA into five other report-index entries. Actual representation files agree with their own evaluation indices and the statistics sources. The report uses explicit audited run paths; it does **not** claim the historical index or source code was patched in this task.
- No human semantic review took place. Same-class donors and shams do not establish semantic validity.
- The output-only detector means uncertainty plus entropy, not every possible output feature. RQ3 populations are clean-correct and non-sham, separately for each model/seed.
- Raw EuroSAT images, pretrained binaries, transformation caches, licensing and external-resource publication are not certified complete by this document audit. Full experiment reproduction requires them or verified retrieval/regeneration instructions; document reproduction does not.
- No training, final-test inference, Git commit, tag creation, push or slides were performed. The local `sac-campaign-v1-final` tag is distinct from the course-required `v1.0-final`. The report cites the supplied repository URL, but anonymous access still needs owner verification.
- The user confirmed the affiliation. That answer did not independently confirm the course's permission for a five-member team; this administrative requirement remains open.
