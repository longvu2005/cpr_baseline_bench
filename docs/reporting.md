# Rebuild benchmark tables

From the repository root:

```bash
python run_baseline.py --tables-only
# Equivalent table builder:
python build_tables.py
# Import preflight only; no manifests, models or evaluated outputs needed:
python -S build_tables.py --check
```

`--tables-only` reads existing evaluated outputs and the ordered manifest hashes.
It does not discover methods, validate/repair image mounts, install requirements,
load encoders, regenerate detections, alter run metadata or reevaluate scores.
The reporting entry points and their dependencies use only the standard library.
Normal single-method and `--all` runs check reporting imports once before gallery
or model work, so a missing reporting module fails early.

The `--list`, `--all` and `--tables-only` modes are mutually exclusive. Tables-only
also rejects a positional method. `--skip-install` is unnecessary for tables-only.

## Inputs and comparability

Each accepted result is a pair of `outputs/<method>/metrics.json` and `run.json`.
The overall CSV includes Macro-Full-mAP, the arithmetic mean of the
per-case Full-mAP values over the cases present on test. Full-mAP remains
query-weighted. The case CSV reports ID and Full mAP and R@1 in RCR order.
Empty cases have blank cells. Values in the CSVs are percentages.

- Metrics must use the RCR test protocol and match the current test
  gallery/query hashes. Tuned runs must also match the current val fingerprint.
- Evaluation records the score-file SHA256 and a canonical digest of the evaluated
  run metadata in metrics.json. Tables reject mismatched pairs, invalid metric
  values, inconsistent case counts and incorrect macro/query-weighted aggregates.
  Unbound historical metrics are skipped; run the current inference/evaluation
  pipeline to regenerate trustworthy artifacts.
- Method IDs must agree between the directory, metrics and adjacent run metadata.
  The inference metadata records the manifests used to score the test set.
- Legacy person results and benchmark scene CLIP results from older checkpoints
  are skipped. A current person result must record the RCR detection cache ID
  and index SHA256, matching manifests and selector provenance, and checksums
  for gallery image bytes, projected boxes and selected memberships.
- Localization must link back to the same detection fingerprint and array digest.
  Across person runs, detector geometry and selected memberships must agree.
  Image-content hashes are required and must agree across scene and person
  methods. These are consistency checks on recorded provenance, not a new model
  forward or a fresh scan of gallery image files.
- The separate text-free diagnostic is excluded from the primary table.

Person results using RCR detections carry the label `Shared RCR boxes`. Scene results
carry `Native scene v1`. Historical metrics and model scoring are not rewritten.
If comparison fails, rerun the affected baseline with the current manifests and
shared cache rather than editing provenance to force acceptance.

All results and both row sets are validated/formatted before CSV writes begin.
Validation or formatting failures preserve completed tables. Each CSV replacement
is atomic, so an interrupted write preserves that file. The two CSV replacements
are independent; this is not a transaction across both output files.

## Verification

```bash
python -m pytest -q tests/test_table_reporting.py
python -m pytest -q
```

Regression tests exercise standalone `python -S` entry points without adapters,
models or a method environment, provenance mismatches, current scene/person
comparisons, early preflight, and preservation of existing tables after failures.
