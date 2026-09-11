# SAC-QUTAB Review Closure

> Historical review snapshot through 2026-09-04. References to the academy allocation describe the then-current target and are obsolete for execution. See [Training Readiness](training_readiness.md) for the current verdict.

## Stage VI disposition

The already completed post-implementation scientific review was not repeated. Its frozen B-01–B-05 and H-06–H-08 findings are closed in `docs/SCIENTIFIC_CORRECTIONS_CLOSURE.md`. Exactly one bounded independent verification pass was then performed by the primary agent on 2026-09-03. No subagent, advisor, or additional reviewer was used, and no new review cycle was opened.

## Required closure items

| VI item | Concrete evidence | Disposition |
|---|---|---|
| Recovered scientific findings resolved | Frozen IDs, severities, criteria, files, commands, and statuses in `docs/SCIENTIFIC_CORRECTIONS_CLOSURE.md`; implementations in `src/sac_qutab/{data,review,governance,models,train,evaluate,detector,statistics,reporting,cli}.py`. | closed |
| One bounded independent verification pass | Checklist below, performed once after corrections. It identified only direct correctness/documentation issues already within B-05, H-06, and H-08; those were fixed before final tests. | closed |
| No additional reviewers | Work remained single-agent as required. | closed |
| Only blocker/high/direct correctness repairs | Changes were limited to full-byte weight authentication, aggregate two-rank preflight, report/config authentication, clustered RQ3 output correctness, acceptance tests, and documentation reconciliation. No model, dataset, metric, RQ, or transformation was added. | closed |
| Affected checks rerun | Initial explicit post-edit focus: 34 passed and two new fixture tests failed; fixture-only corrections were rerun as 10 passed. The final complete suite subsequently passed 39 tests. | closed |
| Complete CPU-safe suite | `C:\Users\fereh\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q -p no:cacheprovider` with `PYTHONPATH=src` — exit 0; 39 passed, 0 failed, 0 skipped, 0 xfailed, 36 warnings. | closed |
| Final bounded implementation scan | Source/tests/config scan for `TODO`, `FIXME`, `NotImplemented`, and incomplete markers; schema/command scan across active docs. Result: zero incomplete markers and zero stale-contract markers. The only bare `pass` is the intentional `MetricUndefinedError` body. | closed |

## Single verification-pass checklist

| Check | Evidence inspected | Result |
|---|---|---|
| Source-to-implementation traceability | Five authoritative sources; `docs/REQUIREMENTS_TRACEABILITY.md`; module/CLI/test inventory. | All committed Track 1 RQ1–RQ3 requirements map to runtime code or an explicit external evidence gate. |
| Split and leakage protection | `validate_split_contract`, pair-role validation, detector fit-role checks, protocol freeze; seed/fraction/role/count mutation tests. | Strict manifest-linked four-way roles and group isolation are enforced; calibration and final roles cannot substitute for one another. |
| Model/preprocessing compatibility | Frozen ResNet18/DeiT-Tiny presets, 224 geometry, ImageNet normalization, strict DeiT adapter and forward-shape test. | Compatible and locally exercised without downloading weights. Exact real files remain gated. |
| Runtime consumption of configuration | Strict dataclass schema plus usage scan of all `cfg.*` surfaces; protocol/report contract comparisons. | Fields affect execution, refusal logic, or immutable provenance; no dead core toggle was found. |
| Transform/semantic preservation | Native-domain transform formulas, nested severity/donor invariants, cache hashes, review display/rules/aggregate recomputation. | Contracts are enforced. Human judgments remain appropriately unavailable. |
| Deterministic seeds | Split, transform, audit, training, bootstrap seeds; epoch/sample augmentation; rank RNG checkpoint/restore. | Deterministic derivations and same-topology resume contract are explicit and tested. |
| Metrics/statistics | Hand probability/calibration tests, clustered contrasts, group-level Wilcoxon, absolute RQ3 AP/AUROC intervals, seed summaries. | Direct H-06 output-shape/adequacy defect corrected; undefined/underpowered states remain explicit. |
| Provenance/artifacts | Full file hashes through split→pairs→review→checkpoint→ECDF/detector→protocol→evaluation index→statistics→report. | Direct B-04/B-05 gaps corrected; mutation/removal is fail-closed. |
| CPU/local failure behavior | CPU timing is ineligible for compute freeze; no dataset/weight download path; active authorization gate; explicit interpreter and sandbox failure evidence. | Local checks fail honestly where hardware/evidence is absent; no training was launched. |
| Fabricated-output absence | Repository artifact inventory and docs/results language inspected. | No EuroSAT metric, trained core checkpoint, human-review result, or A100 result is claimed or synthesized. |

## Verification-pass findings resolved in place

1. Reporting accepted a protocol without comparing its full config-derived metric/statistical contract. `reporting.generate_report_artifacts` now requires `Config`, exact six-run keys, condition signature, and metric/statistics contracts; CLI call sites and tests were updated.
2. Absolute RQ3 rows used `effect/ci` while the report consumed `estimate/lower/upper`, and descriptive scores were suppressed when learned detectors were unavailable. Statistics now emits report-compatible bounds and per-score bootstrap state/seed independently of learned-model adequacy.
3. Documentation still contained nonexistent CLI options, incorrect initialization/reviewer-order wording, under-length compute admission, per-rank cap semantics, and old schema/artifact names. README, runbook, rationale, traceability, plan, and validation closure were reconciled.

No blocker or high-severity finding remains unresolved. External A100/data/weight/human-review evidence is deferred with commands and criteria; it is not treated as incomplete code.

## 2026-09-04 MIG allocation addendum

No second scientific-review cycle was opened. The user-reported compute allocation changed from the earlier assumed multi-GPU topology to one future 20 GB A100 MIG instance, while A100 access remains unavailable. The local update migrated administrative/preflight schemas to v2, made CUDA selectors opaque, bound Team 10 admission to world size one and an 18 GiB operational cap, included bounded validation in time projection, aligned dry-run loader behavior with production, and updated refusal tests/documentation. The affected set passed 25 tests; static compilation and the world-size-one six-run matrix passed; the complete CPU-safe suite ran once and passed 47 tests with 36 known fixture warnings. These changes do not alter any research question, model, dataset, transformation, metric, seed, or frozen finding status. Hardware identity, performance, memory, worker selection, resume, and training remain deferred without fabricated evidence.
