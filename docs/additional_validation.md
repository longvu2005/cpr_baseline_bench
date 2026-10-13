# Validation of the seven-method update

Base: `ca00427fbb13e3ad68cc0449c6e4202557641588`.
Validation date: 2026-10-13. Environment: Python 3.12, PyTorch 2.5.1 CPU,
torchvision 0.20.1 CPU, Transformers 4.51.3, Sentence Transformers 4.1.0.

## Completed checks

| Check | Result | Scope |
| --- | --- | --- |
| Original test suite before editing | 136 passed | Establishes original CPU protocol behavior |
| Final suite with downloaded author modules/released mappers | 157 passed, one upstream deprecation warning | Original 136, 18 new adapter/contract tests, 3 native conformance tests |
| All seven requirements | pip dry-run succeeded | Dependency resolution in the validation environment; no GPU inference claim |
| Compilation and pyflakes | Passed | New package/tests/smoke utility and repository compilation |
| Method discovery | All 7 additions discovered | IDs/aliases/config/run/download paths |
| Official SEARLE-XL release | Downloaded and SHA-256 verified | Real Phi weights loaded strictly; pseudo-token forward with a tiny random CLIP backbone |
| Official FTI4CIR release | Downloaded and HF SHA-256 verified | Real serialized IMG2TEXT loaded; native patch/attribute/token-replacement forward with a tiny random CLIP backbone |
| Context-I2W author modules | Native forward passed | Exact custom tokenizer and context mapper with tiny random CLIP/mapper parameters; author checkpoint was not downloaded |
| GME author inference module | Fused forward passed | Tiny random Qwen2VL model, image patches and text input; finite unit-normalized [1,32] embedding |
| Multi-positive evaluation integration | Passed | Toy encoder through complete-gallery score writer, unchanged evaluator and table builder |
| Public model input allowlist | Passed | Poisoned GT labels/positives/case/target fields do not change model input views |
| Shared membership and Partial SetMatch | Passed | Real shared adapter on a deterministic test detector/selector; group expansion and padded missing-member slots |
| Cache failures/interruption | Passed | Feature corruption rejected/rebuilt; completed caption records reused; failed requests not accepted as output |
| Prepared asset integrity | Passed | Changed checkpoint bytes are rejected |
| Runtime separation | Passed | Captioner requires verified shared caches; --skip-install cannot install missing grounding dependencies |
| Frozen benchmark components | Byte-identical to base | evaluate.py, prepare_data.py, benchmark_data.py, protocol.yaml, detection.py, localization.py and matching.py unchanged |
| Existing native architecture signatures | Passed | Original encoder/composition/score functions protected by existing tests |

SEARLE-XL release SHA-256:
`0ce31fb8eecaad95cdf60287dbad2ac9f968d79b74559f11b462701ce4693b95`.

FTI4CIR release SHA-256:
`aa1acd03fba4e70f77b6afbf296f849e58dddcb899f05e613bf677435ff1a967`.

## Not completed in this environment

No end-to-end pretrained GPU benchmark was run. The actual gallery image
files and CUDA were unavailable. SigLIP2, Qwen, MPNet and GME full pretrained
weight inference, the Context-I2W released checkpoint, and a cold full-gallery
GroundingDINO preprocessing run remain to be checked in the user's runtime.
The native conformance tests above use real author code and (where stated)
real mapper weights, with small random backbones; they are not full pretrained
accuracy tests.

The ZIP contains no fabricated benchmark scores, metrics, gallery images or
model weights. Prepare actual assets and use the included native smoke script
before long runs. Optional native tests skip when their downloaded source or
weights are absent. No accuracy or GPU memory/runtime claim is made.

See [integration and run instructions](additional_baselines.md) for asset
locations, cache preparation, source revisions and protocol limitations.
