# FAFA checkpoint

Use the **official released FAFA/SynCPR weight** from the "Pre-trained Model"
release documented by `FAFA_SynCPR/README.md` in
`Delong-liu-bupt/Composed_Person_Retrieval`.

Expected checkpoint file:

```text
checkpoints/fafa/tuned_recall_at1_step.pt
```

The benchmark adapter pins the official source repository to commit
`0cc16936f031f7ad166be4cce1be33d0b44b728e` and does not train or tune FAFA on
the CPR benchmark data.

Do not commit checkpoint files.
