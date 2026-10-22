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
The evaluator's metric values and the existing percentage formatting are retained.

- Metrics must match the current gallery/query manifest hashes. Previous-data
  results are skipped, rather than mixed into the current comparison.
- Method IDs must agree between the directory, metrics and adjacent run metadata.
  An optional runner manifest stamp must match the evaluated stamp. Scene results
  produced by the direct evaluator can omit this runner stamp.
- Legacy person-protocol results are skipped. A current person-protocol result
  must have complete detector/localization fingerprints, matching versions and
  manifest hashes, SHA256 digests for image bytes, actual detection arrays and
  selected memberships, and selector provenance.
- Localization must link back to the same detection fingerprint and array digest.
  Across person runs, detector geometry and selected memberships must agree.
  Image-content hashes must agree across runs that record them, including scene
  methods. These are consistency checks on recorded provenance, not a new model
  forward or a fresh scan of gallery image files.
- The separate text-free diagnostic is excluded from the primary table.

Shared HF results are labeled `Shared person HF v2`; scene results retain
`Native scene v1`. Historical metrics and model scoring are not rewritten.
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
Detector, FAFA text policy, evaluator and retrieval algorithms are unchanged by
the reporting follow-up.
