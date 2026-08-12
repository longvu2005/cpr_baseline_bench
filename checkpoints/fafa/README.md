# FAFA checkpoint

Place the **official released FAFA/SynCPR weight** at:

```text
checkpoints/fafa/tuned_recall_at1_step.pt
```

Source: the "Pre-trained Model" link in `FAFA_SynCPR/README.md` of
`Delong-liu-bupt/Composed_Person_Retrieval`.

Official Google Drive file ID:

```text
1Bf2Ia7zmxx5k3Dj-nRr3CLbAqc_zkM0y
```

Example download command:

```bash
python -m pip install gdown
gdown 1Bf2Ia7zmxx5k3Dj-nRr3CLbAqc_zkM0y \
  -O checkpoints/fafa/tuned_recall_at1_step.pt
```

Do not commit the model weight. The benchmark adapter pins the official source
repository to commit `0cc16936f031f7ad166be4cce1be33d0b44b728e` and does not
train or tune FAFA on the CPR benchmark data.
