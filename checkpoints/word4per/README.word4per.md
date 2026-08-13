# Word4Per checkpoint

This directory stores the Stage-2 Word4Per artifacts used by the `word4per_setmatch` benchmark adapter.

Checkpoint status:

```text
REPRODUCED
```

The public authors' `old_project` documentation exposes a Stage-1 model but does not document a public final Stage-2 `best.pth` download. The benchmark therefore expects a Stage-2 model reproduced from the authors' training recipe on CUHK-PEDES, without using CPR benchmark data for training, tuning, or checkpoint selection.

Expected files:

```text
checkpoints/word4per/word4per_cuhk_pedes_stage2_best.pth
checkpoints/word4per/word4per_cuhk_pedes_stage2_configs.yaml
```

The matching Stage-2 `configs.yaml` must come from the same reproduced experiment as `best.pth`.

Checkpoint preparation/validation is handled by:

```text
methods/published/01_word4per_setmatch/download_checkpoint.py
```

Because no documented official final Stage-2 download is available, that script validates the reproduced artifacts and the documented Stage-2 recipe, then prepares the pinned official source and public auxiliary CLIP/detector weights. It fails clearly when the reproduced pair is missing rather than pretending to download an official final checkpoint.

Do not commit dataset or checkpoint files.
