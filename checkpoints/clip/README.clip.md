## OpenAI CLIP ViT-B/16

Model:  
OpenAI CLIP ViT-B/16.

Official source:  
https://github.com/openai/CLIP

Paper:  
*Learning Transferable Visual Models From Natural Language Supervision*, arXiv:2103.00020.

Checkpoint availability:

- The official OpenAI CLIP repository provides the pretrained ViT-B/16 checkpoint.
- The checkpoint is available through an official direct download URL.
- The benchmark uses the original OpenAI pretrained checkpoint.
- No training, fine-tuning, or hyperparameter tuning is performed on the CPR pilot.

Benchmark policy:  
Use the original OpenAI CLIP ViT-B/16 pretrained checkpoint directly for inference on the CPR benchmark.

Do not train or tune on the CPR pilot.

### Download

Download the official OpenAI CLIP ViT-B/16 checkpoint:

```bash
mkdir -p checkpoints/clip

wget \
    https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
    -O checkpoints/clip/ViT-B-16.pt
```

Expected checkpoint file:

```text
checkpoints/clip/ViT-B-16.pt
```

Do not commit checkpoint files.