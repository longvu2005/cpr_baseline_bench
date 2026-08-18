#!/usr/bin/env python3
"""P10 Imagine-and-Seek: paper-faithful LDRE-L + IP-CIR CPR adapter.

The root benchmark invokes this file with Kaggle's system Python. Before importing
legacy ML dependencies, this script re-execs itself inside the P10 Python-3.10
environment prepared by download_checkpoint.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config.yaml"


def _bootstrap_isolated() -> None:
    if os.environ.get("IP_CIR_ISOLATED") == "1":
        return
    # PyYAML is deliberately one of the tiny benchmark-side dependencies.
    import yaml
    config_arg = None
    for i, token in enumerate(sys.argv[1:]):
        if token == "--config" and i + 2 <= len(sys.argv[1:]):
            config_arg = sys.argv[1:][i + 1]
            break
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    py = Path(str(cfg["isolated_env"]["python"]))
    if not py.is_absolute():
        py = (ROOT / py).resolve()
    if not py.is_file():
        raise SystemExit(
            f"P10 isolated environment is missing: {py}. "
            "Run `python run_baseline.py imagine_seek` without --skip-install so phase 3 can prepare it."
        )
    env = os.environ.copy()
    env["IP_CIR_ISOLATED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    os.execve(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_bootstrap_isolated()

# Heavy imports only after re-exec.
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import clip  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark_progress import PhaseTracker, progress_bar  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

METHOD_ID = "imagine_seek"
ADAPTER_VERSION = "2026-08-19-v5-ldre-l-mounted-assets-streamed-proxies"
FEATURE_SCHEMA = 5


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{lineno}: expected object")
            rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_gallery_index(gallery: Sequence[dict[str, Any]]) -> dict[Any, int]:
    result = {}
    for gi, row in enumerate(gallery):
        image_id = row.get("image_id")
        if image_id in result:
            raise ValueError(f"Duplicate gallery image_id {image_id!r}")
        result[image_id] = gi
    return result


def query_indices(queries: Sequence[dict[str, Any]], gallery_index: dict[Any, int]) -> np.ndarray:
    values = []
    for qi, q in enumerate(queries):
        image_id = q.get("image_id")
        if image_id not in gallery_index:
            raise ValueError(f"Query {qi} image_id={image_id!r} absent from gallery")
        values.append(gallery_index[image_id])
    return np.asarray(values, dtype=np.int64)


def gallery_path(row: dict[str, Any], gi: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {gi} missing path")
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class TargetPad:
    """Exact target-pad behavior from released src/data_utils.py."""
    def __init__(self, target_ratio: float, size: int):
        self.target_ratio = float(target_ratio)
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(w, h) / self.target_ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        return TF.pad(image, [hp, vp, hp, vp], 0, "constant")


def targetpad_transform(dim: int):
    return Compose([
        TargetPad(1.25, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        lambda im: im.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class PathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform):
        self.paths = list(paths)
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def cache_meta_ok(path: Path, meta_path: Path, expected_meta: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray | None:
    if not path.is_file() or read_json(meta_path) != expected_meta:
        return None
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if arr.shape != shape or arr.dtype != np.float32:
        return None
    # Validate a representative sample; full finite validation happens before scoring output.
    sample = np.asarray(arr.reshape(-1, shape[-1])[: min(64, max(1, arr.size // shape[-1]))])
    if not np.isfinite(sample).all():
        return None
    print(f"[cache] {rel(path)}", flush=True)
    return arr


@torch.no_grad()
def encode_images(
    *, paths: Sequence[Path], model, transform, device: torch.device, batch_size: int,
    out: Path, meta_path: Path, meta: dict[str, Any], dim: int, desc: str,
) -> np.ndarray:
    cached = cache_meta_ok(out, meta_path, meta, (len(paths), dim))
    if cached is not None:
        return cached
    loader = DataLoader(
        PathDataset(paths, transform), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(paths), dim))
    cursor = 0
    for images in progress_bar(loader, desc=desc, total=len(loader), unit="batch"):
        images = images.to(device, non_blocking=True)
        features = F.normalize(model.encode_image(images).float(), dim=-1)
        n = features.shape[0]
        mmap[cursor:cursor+n] = features.cpu().numpy()
        cursor += n
    mmap.flush()
    write_json(meta_path, meta)
    return np.load(out, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def encode_text_pairs(
    *, rows: Sequence[dict[str, Any]], model, device: torch.device, ncap: int, dim: int,
    out: Path, meta_path: Path, meta: dict[str, Any], batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if out.is_file() and read_json(meta_path) == meta:
        try:
            data = np.load(out, allow_pickle=False)
            a, b = data["original"], data["target"]
            if a.shape == (ncap, len(rows), dim) and b.shape == a.shape and np.isfinite(a).all() and np.isfinite(b).all():
                print(f"[cache] {rel(out)}", flush=True)
                return a, b
        except Exception:
            pass

    original_flat: list[str] = []
    target_flat: list[str] = []
    for qi, row in enumerate(rows):
        original = [str(x).strip() for x in row.get("original_captions", []) if str(x).strip()]
        target = [str(x).strip() for x in row.get("target_captions", []) if str(x).strip()]
        if len(original) != ncap or len(target) != ncap:
            raise ValueError(f"Query {qi}: expected {ncap} original/target captions, got {len(original)}/{len(target)}")
        original_flat.extend(original)
        target_flat.extend(target)

    def encode(flat: Sequence[str], label: str) -> np.ndarray:
        chunks = []
        for start in progress_bar(range(0, len(flat), batch_size), desc=label, total=(len(flat)+batch_size-1)//batch_size, unit="batch"):
            batch = list(flat[start:start+batch_size])
            tokens = clip.tokenize(batch, context_length=77, truncate=True).to(device)
            feat = F.normalize(model.encode_text(tokens).float(), dim=-1)
            chunks.append(feat.cpu().numpy())
        q_n_d = np.concatenate(chunks, axis=0).reshape(len(rows), ncap, dim)
        return np.transpose(q_n_d, (1, 0, 2)).astype(np.float32, copy=False)

    original = encode(original_flat, "IP-CIR encode original captions")
    target = encode(target_flat, "IP-CIR encode edited captions")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, original=original, target=target)
    write_json(meta_path, meta)
    return original, target


def validate_proxy_manifest(cfg: dict[str, Any], queries: Sequence[dict[str, Any]]) -> tuple[Path, list[dict[str, Any]]]:
    manifest = resolve_path(str(cfg["proxy"]["manifest"]))
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = load_jsonl(manifest)
    count = int(cfg["proxy"]["count_per_query"])
    if len(rows) != len(queries):
        raise ValueError(f"Proxy manifest rows={len(rows)}, expected={len(queries)}")
    for qi, row in enumerate(rows):
        if row.get("query_index") != qi or row.get("image_id") != queries[qi].get("image_id"):
            raise ValueError(f"Proxy manifest alignment mismatch at {qi}")
        if int(row.get("proxy_count", -1)) != count:
            raise ValueError(f"Proxy count mismatch at {qi}")
        if int(row.get("proxy_feature_index", -1)) != qi:
            raise ValueError(f"Proxy feature index mismatch at {qi}")
        if row.get("storage_mode") != "stream_generate_encode_discard":
            raise ValueError(f"Unexpected proxy storage mode at {qi}: {row.get('storage_mode')!r}")
    return manifest, rows


def prepare_proxies(cfg: dict[str, Any], config_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    script = METHOD_DIR / "prepare_proxies.py"
    subprocess.run([sys.executable, str(script), "--config", str(config_path), "--stage", "all"], cwd=str(ROOT), check=True)
    queries = load_jsonl(resolve_path(str(cfg["data"]["query_manifest"])))
    return validate_proxy_manifest(cfg, queries)


def robust_components(
    *, original: np.ndarray, target: np.ndarray, gallery: np.ndarray,
    source: np.ndarray, proxy_mean: np.ndarray, cfg: dict[str, Any], device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Mirror released retrieval_circo.py debiasing + robust_aug_img equations."""
    r = cfg["retrieval"]
    ncap, nq, dim = target.shape
    gallery_t = torch.from_numpy(np.asarray(gallery)).to(device=device, dtype=torch.float32)
    neg_values = []
    topk = min(int(r["negative_topk"]), gallery_t.shape[0])

    for i in progress_bar(range(ncap), desc="IP-CIR caption debias weights", total=ncap, unit="caption"):
        gt = torch.from_numpy(target[i]).to(device=device, dtype=torch.float32)
        bo = torch.from_numpy(original[i]).to(device=device, dtype=torch.float32)
        # Source code: diff=(target_sim-original_sim), keep only negative entries,
        # negate, top-k per query, then SUM over all queries to obtain one weight/sample.
        diff = gt @ gallery_t.T - bo @ gallery_t.T
        penalty = torch.where(diff < 0, -diff, torch.zeros_like(diff))
        neg_values.append(float(torch.topk(penalty, k=topk, dim=-1).values.sum().item()))
        del gt, bo, diff, penalty

    neg = torch.tensor(neg_values, dtype=torch.float32, device=device)
    maximum = torch.max(neg)
    eps = float(r["denominator_epsilon"])
    if float(maximum) <= eps:
        weights_t = torch.full_like(neg, 1.0 / ncap)
    else:
        weights_t = torch.softmax(neg / maximum / float(r["debiased_temperature"]), dim=0)
    weights = weights_t.cpu().numpy().astype(np.float32)
    debiased = np.tensordot(weights, target, axes=(0, 0)).astype(np.float32, copy=False)
    robust_direction = (target - original).mean(axis=0).astype(np.float32, copy=False)

    # Released code uses scalar max() over the complete feature tensors (not per-query).
    proxy_max = float(np.max(proxy_mean))
    source_max = float(np.max(source))
    direction_max = float(np.max(robust_direction))
    if abs(source_max) < eps or abs(direction_max) < eps:
        raise RuntimeError(
            f"IP-CIR robust scaling denominator too small: source_max={source_max}, direction_max={direction_max}"
        )
    robust = (
        float(r["source_weight"]) * (proxy_max / source_max) * source
        + float(r["semantic_weight"]) * (proxy_max / direction_max) * robust_direction
        + float(r["proxy_weight"]) * proxy_mean
    ).astype(np.float32, copy=False)
    if not np.isfinite(robust).all() or not np.isfinite(debiased).all():
        raise RuntimeError("Non-finite IP-CIR representation")
    return debiased, robust_direction, robust, [float(x) for x in weights]


def validate_scores(scores: np.ndarray, shape: tuple[int, int]) -> None:
    if scores.shape != shape or scores.dtype != np.float32:
        raise ValueError(f"Invalid scores shape/dtype {scores.shape}/{scores.dtype}, expected {shape}/float32")
    for start in range(0, shape[0], 128):
        if not np.isfinite(np.asarray(scores[start:start+128])).all():
            raise ValueError("scores.npy contains NaN/Inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    tracker = PhaseTracker(METHOD_ID, total=7)

    with tracker.phase("Validate isolated environment, manifests, and prepared marker"):
        if not torch.cuda.is_available():
            raise RuntimeError("IP-CIR requires CUDA")
        if not torch.__version__.startswith(str(cfg["isolated_env"]["torch_version"])):
            raise RuntimeError(f"Wrong isolated torch: {torch.__version__}")
        marker_path = resolve_path(str(cfg["migc"]["prepared_marker"]))
        marker = read_json(marker_path)
        if marker is None or marker.get("author_source", {}).get("commit") != cfg["author_source"]["commit"]:
            raise RuntimeError("Missing/stale P10 prepared marker")
        gallery_manifest = resolve_path(str(cfg["data"]["gallery_manifest"]))
        query_manifest = resolve_path(str(cfg["data"]["query_manifest"]))
        gallery = load_jsonl(gallery_manifest)
        queries = load_jsonl(query_manifest)
        gindex = build_gallery_index(gallery)
        qidx = query_indices(queries, gindex)
        device = torch.device(str(cfg["runtime"]["device"]))
        tracker.log(f"gallery={len(gallery):,} queries={len(queries):,} torch={torch.__version__}")

    with tracker.phase("Generate/validate IP-CIR captions, layouts, and five imagined proxies"):
        proxy_manifest, proxy_rows = prepare_proxies(cfg, config_path)
        tracker.log(f"proxy_manifest={rel(proxy_manifest)}")

    with tracker.phase("Load released-default CLIP ViT-L/14 retrieval space"):
        clip_info = marker.get("assets", {}).get("openai_clip", {})
        clip_ckpt = resolve_path(str(clip_info.get("path", "")))
        if not clip_ckpt.is_file():
            raise FileNotFoundError(f"Missing OpenAI CLIP checkpoint from marker: {clip_ckpt}")
        if sha256_file(clip_ckpt) != str(clip_info.get("sha256")):
            raise RuntimeError("OpenAI CLIP checkpoint checksum changed")
        model, _ = clip.load(str(clip_ckpt), device=device, jit=False)
        model.eval()
        dim = int(cfg["retrieval"]["projection_dim"])
        transform = targetpad_transform(int(model.visual.input_resolution))
        if int(model.text_projection.shape[1]) != dim:
            raise RuntimeError(f"Unexpected CLIP projection dim {model.text_projection.shape[1]}")

    with tracker.phase("Encode/cache gallery + source; load streamed proxy CLIP-L features"):
        gallery_paths = [gallery_path(row, gi) for gi, row in enumerate(gallery)]
        clip_sha = sha256_file(clip_ckpt)
        gallery_meta = {
            "schema": FEATURE_SCHEMA, "adapter": ADAPTER_VERSION,
            "manifest_sha256": sha256_file(gallery_manifest), "clip_sha256": clip_sha,
            "preprocess": "targetpad_1.25",
        }
        gallery_features = encode_images(
            paths=gallery_paths, model=model, transform=transform, device=device,
            batch_size=int(cfg["runtime"]["clip_image_batch_size"]),
            out=resolve_path(str(cfg["retrieval"]["gallery_features"])),
            meta_path=resolve_path(str(cfg["retrieval"]["gallery_features_meta"])),
            meta=gallery_meta, dim=dim, desc="IP-CIR encode gallery",
        )
        source = np.asarray(gallery_features[qidx], dtype=np.float32)
        source = source / np.maximum(np.linalg.norm(source, axis=-1, keepdims=True), 1e-12)

        # official_proxy_worker.py encodes each generated proxy immediately in the
        # same released CLIP-L/target-pad feature space, then discards the PNG. This
        # avoids ~15k proxy images consuming Kaggle's 20-GiB writable filesystem.
        proxy_feature_path = resolve_path(str(cfg["retrieval"]["proxy_features"]))
        proxy_state_path = resolve_path(str(cfg["retrieval"]["proxy_features_state"]))
        proxy_state = read_json(proxy_state_path)
        count = int(cfg["proxy"]["count_per_query"])
        if proxy_state is None or proxy_state.get("clip_sha256") != clip_sha:
            raise RuntimeError("Streamed proxy feature state is missing or was encoded with a different CLIP-L checkpoint")
        flat_proxy = np.load(proxy_feature_path, mmap_mode="r", allow_pickle=False)
        expected_proxy_shape = (len(queries), count, dim)
        if flat_proxy.shape != expected_proxy_shape or flat_proxy.dtype != np.float32:
            raise ValueError(
                f"Streamed proxy features {flat_proxy.shape}/{flat_proxy.dtype}, expected {expected_proxy_shape}/float32"
            )
        if not np.isfinite(np.asarray(flat_proxy)).all():
            raise RuntimeError("Streamed proxy features contain NaN/Inf or incomplete rows")
        proxy_mean = np.asarray(flat_proxy).mean(axis=1).astype(np.float32, copy=False)

    with tracker.phase("Encode 15 paired LDRE captions and reproduce released debiasing"):
        ncap = int(cfg["retrieval"]["nums_caption"])
        text_meta = {
            "schema": FEATURE_SCHEMA, "adapter": ADAPTER_VERSION,
            "proxy_manifest_sha256": sha256_file(proxy_manifest), "clip_sha256": clip_sha,
            "nums_caption": ncap,
        }
        original, target = encode_text_pairs(
            rows=proxy_rows, model=model, device=device, ncap=ncap, dim=dim,
            out=resolve_path(str(cfg["retrieval"]["text_features"])),
            meta_path=resolve_path(str(cfg["retrieval"]["text_features_meta"])),
            meta=text_meta,
        )
        debiased, robust_direction, robust, caption_weights = robust_components(
            original=original, target=target, gallery=np.asarray(gallery_features),
            source=source, proxy_mean=proxy_mean, cfg=cfg, device=device,
        )

    with tracker.phase("Compute complete released IP-CIR balancing scores"):
        out_dir = resolve_path(str(cfg["output"]["dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_path = out_dir / "scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path, mode="w+", dtype=np.float32, shape=(len(queries), len(gallery))
        )
        gallery_t = torch.from_numpy(np.asarray(gallery_features)).to(device=device, dtype=torch.float32)
        lam = float(cfg["fusion"]["lambda_text"])
        batch = int(cfg["runtime"]["score_batch_size"])
        for start in progress_bar(range(0, len(queries), batch), desc="IP-CIR final scores", total=(len(queries)+batch-1)//batch, unit="batch"):
            end = min(start + batch, len(queries))
            text_t = torch.from_numpy(debiased[start:end]).to(device=device, dtype=torch.float32)
            robust_t = torch.from_numpy(robust[start:end]).to(device=device, dtype=torch.float32)
            st = text_t @ gallery_t.T
            sp = robust_t @ gallery_t.T
            sf = lam * st + (1.0 - lam) * sp * st
            scores[start:end] = sf.cpu().numpy()
        scores.flush()
        validate_scores(scores, (len(queries), len(gallery)))

    with tracker.phase("Write transparent benchmark metadata"):
        payload = {
            "method": cfg["method"],
            "display_name": cfg["display_name"],
            "group": cfg["group"],
            "paper": cfg["paper"],
            "implementation_status": "OFFICIAL_SOURCE_ADAPTED",
            "paper_configuration": "LDRE-L + IP-CIR",
            "adapter_version": ADAPTER_VERSION,
            "training_free": True,
            "cpr_supervision": "No",
            "retrieval": {
                "backbone": cfg["retrieval"]["clip_name"],
                "preprocess": "released targetpad ratio 1.25",
                "nums_caption": int(cfg["retrieval"]["nums_caption"]),
                "debiased_temperature": float(cfg["retrieval"]["debiased_temperature"]),
                "caption_weights": caption_weights,
                "source_weight": float(cfg["retrieval"]["source_weight"]),
                "semantic_weight": float(cfg["retrieval"]["semantic_weight"]),
                "proxy_weight": float(cfg["retrieval"]["proxy_weight"]),
                "fusion_weight": float(cfg["fusion"]["lambda_text"]),
                "fusion_source": cfg["fusion"].get("source"),
                "formula": "Sf=lambda*St+(1-lambda)*(Sp*St)",
                "baseline": "LDRE-L",
                "important": "IP-CIR is plug-and-play over a baseline. This adapter reproduces the paper's CLIP-L branch: LDRE-L + IP-CIR; P5 LinCIR-L is intentionally not substituted for the paper's LinCIR-G branch.",
            },
            "generator": {
                "author_repository": cfg["author_source"]["repository"],
                "author_commit": cfg["author_source"]["commit"],
                "captioner": cfg["captioner"]["repo_id"],
                "layout_llm": cfg["layout_llm"]["repo_id"],
                "proxy_count": int(cfg["proxy"]["count_per_query"]),
                "proxy_reference": "full query scene used as ELITE concept mask",
                "proxy_storage": "generate -> CLIP-L targetpad encode -> discard; sparse audit PNGs only",
            },
            "cpr_adapter_boundary": {
                "reason": "CPR lacks author CIRCO shared_concept/object-mask annotations",
                "uses_gt_target_box": False,
                "uses_gt_target_identity": False,
                "uses_cpr_positive_labels": False,
                "uses_query_instruction": True,
                "small_model_fallback": False,
            },
            "num_queries": len(queries),
            "num_gallery": len(gallery),
            "scores": rel(scores_path),
            "higher_is_better": True,
            "query_image_removed_inside_method": False,
            "prepared_marker": rel(marker_path),
        }
        write_json(out_dir / "run.json", payload)
        tracker.log(f"scores={rel(scores_path)}")
        del model, gallery_t
        gc.collect()
        torch.cuda.empty_cache()
    tracker.finish()


if __name__ == "__main__":
    main()
