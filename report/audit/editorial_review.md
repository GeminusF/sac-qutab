# Academic editorial review, 2026-09-11

The pre-edit manuscript is preserved in `editorial-baseline-20260911/`. The review covered every English draft section, appendix prose, and figure captions. Generated LaTeX sections agree with the editable drafts.

## Content-preservation checks

- No distinct numeric literal was added or removed across the draft and appendix sources.
- Every inline and displayed mathematical expression is unchanged within its source file.
- Scientific citation multisets remain unchanged. A sixteenth bibliography item now cites the team-supplied public repository as the reproducibility entry point.
- The prose-only baseline preserved the then-current title block. A later approved presentation revision changed the paper title, acronym expansion, institutional logos and author layout without changing the scientific results.
- The existing quantitative and visual-source checks pass, including all six runs, the frozen headline calculations, the original 15 supplementary CSVs and 47 PNGs, and the retained gallery and attention inputs.
- Source whitespace-delimited word counts, including source markup, were 7,880 before editing and 7,688 after the prose pass. These are comparison counts, not submission word counts. The later title revision leaves the abstract at 221 whitespace-delimited words.

## Language and assembled-document checks

- No authored em dash or TeX em-dash sequence remains.
- The sentence-opening scan finds one `It`, one `This`, and no `We` openings; none are adjacent. Contextual uses were reviewed rather than prohibited mechanically.
- Content-bearing wording such as `additionally requires` is retained. The automated transition-word check targets sentence-initial filler, not every occurrence of a word.
- Tectonic successfully compiled the revised manuscript. All 21 PDF pages were rendered at 130 dpi and visually inspected, with additional full-page checks of equations and references.
- The main report including references occupies 12 pages. The 9-page appendix starts on page 13.
- No unresolved reference markers, clipped content, overlapping text, or unreadable required figures were found. All reported PDF fonts are embedded. Non-blocking font-shape and underfull-box warnings are recorded in `pdf_qa.json`.
- `validate_package.py` passes all 500 source and PDF checks.

The current reviewed PDF hash is recorded in `pdf_qa.json`; the pre-presentation-revision PDF remains preserved in the dated baseline directory.

This editorial review does not replace the experimental source audit, certify public release availability, or resolve the author and course-administration checks listed in `qa_status.md`.
