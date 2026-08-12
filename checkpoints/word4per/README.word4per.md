## Per2Word (Pic2Word-based)

Model:
Per2Word, benchmark adaptation based on Pic2Word / Mapping Pictures to Words for Zero-shot Composed Image Retrieval

Official source:
https://github.com/google-research/composed_image_retrieval

Paper:
*Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval*, CVPR 2023, arXiv:2302.03084.

Checkpoint availability:

- The official Pic2Word repository provides a pretrained checkpoint through Google Drive.
- The released model uses the OpenAI CLIP ViT-L/14 backbone.
- The Pic2Word checkpoint provides the learned image-to-word mapping network (`img2text`).
- The official Google Drive checkpoint may be unavailable through `gdown` because of Google Drive permissions or quota restrictions.
- The benchmark therefore uses the official OpenAI CLIP ViT-L/14 checkpoint together with the released Pic2Word `img2text` weights.
- When the official Google Drive file is inaccessible, the public NAVER Hugging Face mirror is used as a fallback for the Pic2Word mapper.
- The NAVER file is a third-party mirror and should not be described as an official checkpoint source.
- No training, fine-tuning, or hyperparameter tuning is performed on the CPR pilot.

Benchmark policy:
Use the pretrained Pic2Word mapper together with the official OpenAI CLIP ViT-L/14 backbone.

Do not train or tune on the CPR pilot.

Download:

```bash
mkdir -p checkpoints/clip
mkdir -p checkpoints/per2word

wget \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
  -O checkpoints/clip/ViT-L-14.pt

wget \
  https://huggingface.co/navervision/zeroshot-cir-models/resolve/main/pic2word_large.pt \
  -O checkpoints/per2word/pic2word_vitl14.pt