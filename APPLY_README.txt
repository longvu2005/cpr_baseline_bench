Apply the complete update at the root of the existing cpr_baseline_bench checkout.
Base commit: ca00427fbb13e3ad68cc0449c6e4202557641588

1. Extract cpr-baselines-update.zip at the repository root.
2. Read docs/additional_baselines.md and docs/additional_validation.md.
3. Run: python run_baseline.py --list
4. Run a selected method: python run_baseline.py siglip2_compose

UPDATE_MANIFEST.json records every supplied file and original hashes for replacements.
The evaluator, data manifests, data preparation and matching formulas are unchanged.
Weights/images/caches are not bundled. Downloads and native smoke commands are in the guide.
The list supplied by the user has seven method names. No eighth was fabricated.
