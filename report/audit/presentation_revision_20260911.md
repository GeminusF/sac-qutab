> Historical report-build record. The publication uses canonical assets in `report/figures/` and `GeminusF/sac-qutab`; old paths below describe the earlier build.

# Title, author and table-presentation revision, 2026-09-11

This revision changes the manuscript presentation layer only. Experiment schemas, saved metrics, model outputs, figures and numeric CSV sources are unchanged.

## First page

- Paper title: `Semantic Analysis of Counterfactuals: Quality Under Transformations and Adaptive Benchmarking`.
- PDF metadata uses the same title.
- The institutional bar contains AZCON, NAIC and AI Academy, followed by a 0.4 pt horizontal rule.
- The SAC-QUTAB logo appears in the title row, immediately left of the two-line title.
- NAIC source and compiled wordmark SHA-256: `3f409de0850be54ff2d6c85be4f702124a441ddf1acac7ed2004982920511421`.
- AI Academy source and compiled wordmark SHA-256: `5fa6cc4c557dd768670c210178371d3763ee11589194c392dd83b1945d8ad084`.

## Authors

The visible 3+2 grid and PDF metadata use the same contribution order:

1. Milana Karimova
2. Sharaf Feyzullayev
3. Farah Feyzullayev
4. Jeyhuna Sevdiyeva
5. Nicat Aghayev

Each visible author block contains the supplied email address and the two-line AI Academy / National Intelligence Center of Azerbaijan affiliation. Author-reference marks and the former shared affiliation line were removed.

## Abstract and references

The SAC-QUTAB acronym is expanded in the Abstract and Introduction. The 221-word Abstract contains no em dash and cites the repository without printing its URL inline. Reference 16 records `https://github.com/GeminusF/dl-final-project`. Anonymous public access and the requested `v1.0-final` tag remain owner checks.

## Performance emphasis

Bold formatting follows metric direction and comparison scope rather than numeric magnitude alone:

- clean performance: 5 cells;
- severe-minus-mild contrasts: 10 cells, including the displayed amplitude-mixing ECDF tie;
- absolute detector scores: 4 cells;
- per-seed clean results: 5 cells;
- full-condition summaries: 38 cells, including displayed ties;
- detector gains: 4 gain cells; confidence-interval bounds remain unbolded.

Non-performance and population tables contain no added bold metric cells. SHA-256 checks confirm that all nine table CSV sources match the pre-revision baseline.

## Build and QA

- Compiler: Tectonic 0.17.0.
- PDF: `report.pdf`.
- PDF SHA-256: `a7580b4df073bcf2fba5dcff89842d12183ac13a8642968aa2a4ae0490e0d6db`.
- Total pages: 27.
- Main report including references: pages 1-18.
- Appendix: pages 19-27.
- All pages were rendered at 110 dpi and visually inspected; Figure 1 also received an original-resolution page check.
- No unresolved reference markers, clipped content, overlapping blocks or unembedded fonts were found.
- Global exact placement intentionally increases the main report length and leaves white space before some full-width visuals.

## Global exact-placement revision

- Ordinary figures and tables use `[H]`.
- Eleven full-width main-text visuals use non-floating page-top blocks in source order.
- The approved Figure 1 source is `report_materials/pipeline_diagram_vibrant.pdf`; its canonical copy has SHA-256 `fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71`.
- An initial `cuted` implementation was rejected during visual QA because it split Figure 1, omitted intervening text and introduced stray equals-sign artifacts. The final page-top implementation preserves all text and captions.
