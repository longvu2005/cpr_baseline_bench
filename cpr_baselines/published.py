"""Adapters calling the official architectures and native pseudo-token functions."""
from __future__ import annotations

import importlib
import sys

from cpr_person.cache import resolve
from .backends import dtype_for
from .common import load_source_file


def torch_load(path):
    import torch
    # FTI4CIR's author release contains a serialized IMG2TEXT object, not only
    # tensors. This is deliberately restricted to validated pinned assets.
    return torch.load(path, map_location="cpu", weights_only=False)


def strip_module(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a non-empty official state dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def openai_clip(cfg, device):
    source = resolve(cfg["assets"]["clip_code"]["path"])
    sys.path.insert(0, str(source))
    clip = importlib.import_module("clip")
    if not resolve(clip.__file__).is_relative_to(source):
        raise ImportError("Another installed CLIP package is shadowing the pinned OpenAI source")
    model, preprocess = clip.load(str(resolve(cfg["assets"]["clip"]["path"])), device=device, jit=False)
    model.to(device=device, dtype=dtype_for(cfg, device)).eval()
    return clip, model, preprocess


def targetpad(preprocess, ratio):
    from torchvision.transforms import Compose
    from torchvision.transforms.functional import pad
    # Same integer padding and CLIP resize/crop as the authors' TargetPad.
    def padding(image):
        width, height = image.size
        if max(width, height) / min(width, height) < ratio:
            return image
        scaled = max(width, height) / ratio
        return pad(image, [max(int((scaled - width) / 2), 0),
                           max(int((scaled - height) / 2), 0)] * 2, 0, "constant")
    return Compose([padding, preprocess])


class SEARLE:
    def __init__(self, cfg, device):
        import torch
        self.torch, self.device, self.cfg = torch, device, cfg
        self.clip, self.model, preprocess = openai_clip(cfg, device)
        self.preprocess = targetpad(preprocess, float(cfg["composition"]["targetpad_ratio"]))
        source = resolve(cfg["source"]["local_checkout"]) / "src"
        phi_module = load_source_file("cpr_searle_phi", source / "phi.py")
        self.pseudo_encoder = load_source_file("cpr_searle_pseudo", source / "encode_with_pseudo_tokens.py").encode_with_pseudo_tokens
        self.phi = phi_module.Phi(input_dim=768, hidden_dim=3072, output_dim=768, dropout=0.5)
        raw = torch_load(resolve(cfg["assets"]["checkpoint"]["path"]))
        if not isinstance(raw, dict) or "Phi" not in raw:
            raise ValueError("Official SEARLE-XL release must contain Phi")
        self.phi.load_state_dict(raw["Phi"], strict=True)
        self.phi.to(device=device, dtype=self.model.dtype).eval()

    def pixels(self, images):
        return self.torch.stack([self.preprocess(image) for image in images]).to(self.device, self.model.dtype)

    def images(self, images):
        with self.torch.inference_mode():
            return self.model.encode_image(self.pixels(images)).float().cpu().numpy()

    def queries(self, images, texts, reference_features=None):
        if any("$" in text for text in texts):
            raise ValueError("Query contains SEARLE's reserved '$' placeholder")
        template = self.cfg["composition"]["prompt_template"]
        if template.count("$") != 1 or template.count("{text}") != 1:
            raise ValueError("SEARLE prompt needs exactly one '$' and one '{text}' placeholder")
        prompts = [template.format(text=text) for text in texts]
        tokens = self.clip.tokenize(prompts, truncate=True).to(self.device)
        if not self.torch.all((tokens == 259).sum(dim=1) == 1):
            raise ValueError("SEARLE requires exactly one pseudo-token placeholder per prompt")
        with self.torch.inference_mode():
            # Phi was trained on raw CLIP image features; normalizing before Phi
            # would change the published model.
            pseudo = self.phi(self.model.encode_image(self.pixels(images)))
            return self.pseudo_encoder(self.model, tokens, pseudo).float().cpu().numpy()


class ContextI2W:
    def __init__(self, cfg, device):
        import torch
        self.torch, self.device, self.cfg = torch, device, cfg
        source = resolve(cfg["source"]["local_checkout"])
        sys.path.insert(0, str(source))
        official_clip = importlib.import_module("model.clip")
        official_model = importlib.import_module("model.model")
        tokenizer_module = importlib.import_module("third_party.open_clip.simple_tokenizer")
        for module in (official_clip, official_model, tokenizer_module):
            if not resolve(module.__file__).is_relative_to(source):
                raise ImportError("Another package is shadowing pinned Context-I2W modules")
        self.tokenizer = tokenizer_module.SimpleTokenizer()
        self.model, _, self.preprocess = official_clip.load(str(resolve(cfg["assets"]["clip"]["path"])),
                                                           device="cpu", jit=False)
        self.mapper = official_model.IM2TEXT_OWM_Attn(transformer_width=self.model.transformer_width,
            vision_width=self.model.vision_width, embed_dim=self.model.embed_dim, middle_dim=512,
            output_dim=self.model.token_embedding.weight.shape[1], n_layer=2)
        raw = torch_load(resolve(cfg["assets"]["checkpoint"]["path"]))
        if not isinstance(raw, dict) or not {"state_dict", "state_dict_img2text"} <= raw.keys():
            raise ValueError("Context-I2W requires both official CLIP and IM2TEXT_OWM_Attn state dicts")
        self.model.load_state_dict(strip_module(raw["state_dict"]), strict=True)
        self.mapper.load_state_dict(strip_module(raw["state_dict_img2text"]), strict=True)
        dtype = dtype_for(cfg, device)
        self.model.to(device=device, dtype=dtype).eval()
        self.mapper.to(device=device, dtype=dtype).eval()
        self.replace_id = self.tokenizer.encoder["<|replace|>"]
        if self.replace_id != self.model.replace_id:
            raise ValueError("Official Context-I2W tokenizer and CLIP vocabulary differ")

    def pixels(self, images):
        return self.torch.stack([self.preprocess(image) for image in images]).to(self.device, self.model.dtype)

    def tokenize(self, prompts):
        tok = self.tokenizer
        sot, eos = tok.encoder["<|startoftext|>"], tok.encoder["<|endoftext|>"]
        result = self.torch.zeros(len(prompts), 77, dtype=self.torch.long, device=self.device)
        for i, text in enumerate(prompts):
            # The author's custom vocabulary reserves <|replace|> at 49405.
            # Stock OpenAI CLIP tokenization is not compatible with this mapper.
            tokens = [sot] + tok.encode(text)[:75] + [eos]
            result[i, :len(tokens)] = self.torch.tensor(tokens, device=self.device)
        return result

    def images(self, images):
        with self.torch.inference_mode():
            return self.model.encode_image(self.pixels(images)).float().cpu().numpy()

    def queries(self, images, texts, reference_features=None):
        if any("<|replace|>" in text or "<|endoftext|>" in text or "<|startoftext|>" in text for text in texts):
            raise ValueError("Reserved Context-I2W token in query text")
        template = self.cfg["composition"]["prompt_template"]
        if template.count("<|replace|>") != 1 or template.count("{text}") != 1:
            raise ValueError("Context-I2W prompt needs one replacement token and one text field")
        tokens = self.tokenize([template.format(text=text) for text in texts])
        if not self.torch.all((tokens == self.replace_id).sum(dim=1) == 1):
            raise ValueError("Context-I2W requires one context pseudo-word per query")
        with self.torch.inference_mode():
            patches = self.model.encode_image_tokens(self.pixels(images))
            context = self.model.encode_text(tokens)
            pseudo = self.mapper(context, patches)
            return self.model.encode_text_img_retrieval(tokens, pseudo, split_ind=self.replace_id,
                                                       repeat=False).float().cpu().numpy()


class FTI4CIR:
    def __init__(self, cfg, device):
        import torch
        self.torch, self.device, self.cfg = torch, device, cfg
        self.clip, self.model, preprocess = openai_clip(cfg, device)
        # The published FTI4CIR test entrypoint runs CLIP and mapper in fp32.
        self.model.float()
        self.preprocess = targetpad(preprocess, float(cfg["composition"]["targetpad_ratio"]))
        source = resolve(cfg["source"]["local_checkout"]) / "src"
        sys.path.insert(0, str(source))
        for name in ("MHTransformer", "dataset", "utils", "i2t_model"):
            module = importlib.import_module(name)
            if not resolve(module.__file__).is_relative_to(source):
                raise ImportError(f"Another package is shadowing pinned FTI4CIR {name}")
        self.utils = sys.modules["utils"]
        mapper = torch_load(resolve(cfg["assets"]["checkpoint"]["path"]))
        if not isinstance(mapper, sys.modules["i2t_model"].IMG2TEXT):
            raise TypeError("FTI4CIR requires the author's serialized IMG2TEXT release")
        self.mapper = mapper.to(device=device, dtype=torch.float32).eval()

    def pixels(self, images):
        return self.torch.stack([self.preprocess(image) for image in images]).to(self.device, self.torch.float32)

    def images(self, images):
        with self.torch.inference_mode():
            return self.model.encode_image(self.pixels(images)).float().cpu().numpy()

    def queries(self, images, texts, reference_features=None):
        if any("*" in text for text in texts):
            raise ValueError("Query contains FTI4CIR's reserved '*' placeholder")
        with self.torch.inference_mode():
            pixels = self.pixels(images)
            global_features = self.model.encode_image(pixels)
            patches = self.utils.get_img_patch_feats(pixels, self.model)
            local_features, counts = self.mapper.get_img_local_attr_feats(global_features, patches)
            subject_tokens = self.mapper.phi_s(global_features)
            attribute_tokens = self.mapper.phi_a(local_features)
            prompts = [f"a photo of * with {'* ' * count}but {text}" for count, text in zip(counts, texts)]
            tokens = self.clip.tokenize(prompts, truncate=True).to(self.device)
            star = int(self.clip.tokenize(["*"])[0, 1])
            if any(int((row == star).sum()) != 1 + count for row, count in zip(tokens, counts)):
                raise ValueError("FTI4CIR placeholder count changed after tokenization")
            # Same native inference as IMG2TEXT.img_to_text. Only the hardcoded
            # .cuda() is replaced by .to(device), so CPU/CUDA device choice works.
            return self.utils.token_replace(counts, tokens, subject_tokens, attribute_tokens,
                                            self.model, 0).float().cpu().numpy()
