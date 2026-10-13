# Qwen Caption-to-Caption

Run from the repository root:

```bash
python run_baseline.py qwen_caption_to_caption
```

Standalone preparation/inference also accepts `--config`:

```bash
python methods/simple/13_qwen_caption_to_caption/download_checkpoint.py
python methods/simple/13_qwen_caption_to_caption/run.py
python evaluate.py --method qwen_caption_to_caption
python build_tables.py
```

See [the integration guide](../../../docs/additional_baselines.md) for the exact protocol, asset requirements, fairness rules and test limits. Weights are downloaded separately; there is no substitute model or fabricated score if an artifact is missing.

## Caption length and failure handling

- Gallery and target descriptions have the same 65-word instruction, with a fixed
  384-token generation ceiling (greedy, no sampling).
- If Qwen hits its token ceiling without EOS, the generator raises a distinct
  length error. The caption store makes **one** deterministic retry with a compact
  40-word instruction appended to the *original* prompt; it never feeds back a
  partial overflow output. This retry uses the same visible image/instruction
  without identity labels or ground-truth target information.
- Format parsing has its own single retry. Only complete validated captions are
  saved, and exhausted retries fail with an informative error rather than an
  invented or truncated caption. Per-image caption caching remains enabled for
  new resumable runs. The changed generation configuration has new cache keys;
  previous 256-token records are not reused.
- `tests/test_caption_length_retry.py` covers retry success, exhaustion,
  independent format repair, and propagation of unrelated errors.
