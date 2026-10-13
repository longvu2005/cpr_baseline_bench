"""Frozen model inference; heavy packages are imported only when needed."""
from __future__ import annotations

import gc

from cpr_person.cache import resolve
from .common import compose_linear, load_source_file


def dtype_for(cfg, device):
    import torch
    if device.type == "cpu":
        return torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[cfg["runtime"]["dtype"]]


def release_model(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SigLIPCompose:
    def __init__(self, cfg, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.cfg, self.device = torch, cfg, device
        self.dtype = dtype_for(cfg, device)
        path = str(resolve(cfg["assets"]["encoder"]["path"]))
        self.model = AutoModel.from_pretrained(path, torch_dtype=self.dtype, local_files_only=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)

    def images(self, images):
        values = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        with self.torch.inference_mode():
            return self.model.get_image_features(pixel_values=values).float().cpu().numpy()

    def queries(self, images, texts, reference_features=None):
        limit = int(self.cfg["composition"]["max_text_tokens"])
        if not 1 <= limit <= self.model.config.text_config.max_position_embeddings:
            raise ValueError("SigLIP2 text limit exceeds the checkpoint's position embedding size")
        tokens = self.processor(text=texts, padding="max_length", max_length=limit,
                                truncation=True, return_tensors="pt")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.torch.inference_mode():
            features = self.model.get_text_features(**tokens).float().cpu().numpy()
        refs = self.images(images) if reference_features is None else reference_features
        return compose_linear(refs, features, float(self.cfg["composition"]["alpha"]))


class GMECompose:
    def __init__(self, cfg, device):
        # Use the author's pinned inference script and fused image/text forward,
        # including last-token pooling. Do not approximate it with CLIP fusion.
        path = resolve(cfg["assets"]["encoder"]["path"])
        official = load_source_file("cpr_gme_official", path / "gme_inference.py")
        self.cfg = cfg
        self.model = official.GmeQwen2VL(model_path=str(path), device=str(device),
            min_image_tokens=int(cfg["composition"]["min_image_tokens"]),
            max_image_tokens=int(cfg["composition"]["max_image_tokens"]),
            max_length=int(cfg["composition"]["max_length"]),
            local_files_only=True, attn_implementation="eager")
        self.model.base.to(device=device, dtype=dtype_for(cfg, device)).eval()
        self.model.default_instruction = cfg["composition"]["gallery_instruction"]

    def images(self, images):
        return self.model.embed(texts=[None] * len(images), images=images,
                                is_query=False).float().cpu().numpy()

    def queries(self, images, texts, reference_features=None):
        return self.model.embed(texts=texts, images=images, is_query=True,
            instruction=self.cfg["composition"]["instruction"]).float().cpu().numpy()


class SentenceEncoder:
    def __init__(self, cfg, device):
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        self.model = SentenceTransformer(str(resolve(cfg["assets"]["text_encoder"]["path"])),
                                         device=str(device), local_files_only=True)
        self.model.eval()
        self.model.float()
        self.model.max_seq_length = int(cfg["text_encoding"]["max_length"])

    def texts(self, texts):
        return self.model.encode(texts, batch_size=int(self.cfg["runtime"]["text_batch_size"]),
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)


class QwenGenerator:
    def __init__(self, cfg, device):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        self.torch, self.device, self.cfg = torch, device, cfg
        self.process_vision_info = process_vision_info
        path = str(resolve(cfg["assets"]["captioner"]["path"]))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path,
            torch_dtype=dtype_for(cfg, device), attn_implementation="eager",
            local_files_only=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True,
            min_pixels=int(cfg["generation"]["min_pixels"]),
            max_pixels=int(cfg["generation"]["max_pixels"]))

    def generate(self, prompt, image=None):
        content = [] if image is None else [{"type": "image", "image": image}]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": self.cfg["generation"]["system_prompt"]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)
        budget = int(self.cfg["generation"]["max_new_tokens"])
        # Greedy decoding, one request at a time, and a fixed pixel budget are
        # identical for reference and gallery. No GT-dependent retry or prompt.
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=budget)
        generated = output[0, inputs.input_ids.shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if len(generated) >= budget and int(generated[-1]) not in eos:
            raise ValueError("Qwen output reached max_new_tokens; increase the fixed budget and rerun")
        return self.processor.decode(generated, skip_special_tokens=True).strip()


def build_backend(cfg, device):
    backend = cfg["backend"]
    if backend == "siglip2":
        return SigLIPCompose(cfg, device)
    if backend == "gme":
        return GMECompose(cfg, device)
    from .published import ContextI2W, FTI4CIR, SEARLE
    return {"searle": SEARLE, "context_i2w": ContextI2W, "fti4cir": FTI4CIR}[backend](cfg, device)
