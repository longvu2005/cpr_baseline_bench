"""Shared CLIP Subject anchors and disjoint one-to-many memberships."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import PROTOCOL_VERSION
from .cache import ROOT, StageCache, cache_lock, read_json, resolve, sha256_file
from .detection import BoxCandidate, crop_person, image_path, protocol_config, read_rgb, detection_array_sha256


def select_subject_members(similarity, candidates, *, threshold=0.20, margin=0.03, max_members=5):
    sim = np.asarray(similarity, dtype=np.float64)
    if (sim.ndim != 2 or not np.isfinite(sim).all() or sim.shape[0] < 1
            or sim.shape[1] < sim.shape[0] or len(candidates) != sim.shape[1]
            or not np.isfinite([threshold, margin]).all() or margin < 0 or max_members < 1):
        raise ValueError("Need finite similarities and at least one candidate per Subject")
    # SciPy is deterministic for this ordered matrix; do not perturb native
    # CLIP scores with epsilon noise, which could change the true optimum.
    rows, cols = linear_sum_assignment(-sim)
    anchors = [-1] * len(sim)
    for row, col in zip(rows, cols):
        anchors[int(row)] = int(col)
    members = [[col] for col in anchors]
    used = set(anchors)
    for col in np.argsort(-sim.max(axis=0), kind="stable"):
        col = int(col)
        if col in used or candidates[col].fallback:
            continue
        eligible = [row for row in range(len(sim)) if len(members[row]) < max_members
                    and sim[row, col] >= threshold and sim[row, col] >= sim[row, anchors[row]] - margin]
        if eligible:
            owner = max(eligible, key=lambda row: (sim[row, col], -row))
            members[owner].append(col)
            used.add(col)
    return members, anchors


def public_query(query):
    subjects = query["subjects"]
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Every query needs public Subject selection descriptions")
    texts = [str(s.get("select_text") or "").strip() for s in subjects]
    if not all(texts):
        raise ValueError("Empty subjects[].select_text")
    return query["image_id"], texts


@dataclass
class Localization:
    members: list
    stats: dict
    fingerprint: dict
    path: object = None
    cache_hit: bool = False

    def boxes(self):
        return [[[tuple(member["box"]) for member in members] for members in subjects]
                for subjects in self.members]

    def flat(self, qi):
        return [(si, member) for si, members in enumerate(self.members[qi]) for member in members]


class ClipSelector:
    def __init__(self, cfg, device):
        import clip
        import torch
        self.clip, self.torch, self.device = clip, torch, device
        self.model, self.preprocess = clip.load(str(resolve(cfg["checkpoint"])), device=device, jit=False)
        self.model.eval()
        if str(device).startswith("cpu"):
            self.model.float()

    def images(self, images):
        with self.torch.no_grad():
            tensors = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
            feat = self.model.encode_image(tensors).float()
            feat /= feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return feat.cpu().numpy()

    def texts(self, texts):
        with self.torch.no_grad():
            tokens = self.clip.tokenize(list(texts), truncate=True).to(self.device)
            feat = self.model.encode_text(tokens).float()
            feat /= feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return feat.cpu().numpy()


def localize_queries(queries, gallery, detections, selector_cfg, selector):
    index = {row["image_id"]: i for i, row in enumerate(gallery)}
    if len(index) != len(gallery):
        raise ValueError("Duplicate gallery image_id")
    candidates_all = detections.candidates()
    result, similarities, counts = [], [], []
    extras = padded = fallback_anchors = shortage_queries = 0
    crop_features, text_features = {}, {}
    for query in queries:
        image_id, texts = public_query(query)
        gi = index[image_id]
        candidates = list(candidates_all[gi])
        start = int(detections.offsets[gi])
        refs = list(range(start, int(detections.offsets[gi + 1])))
        missing = max(0, len(texts) - len(candidates))
        width, height = detections.image_sizes[gi]
        full_box = (0.0, 0.0, float(width), float(height))
        candidates.extend(BoxCandidate(full_box, -1.0, True) for _ in range(missing))
        refs.extend([None] * missing)
        padded += missing
        shortage_queries += bool(missing)
        feature_key = (gi, len(candidates))
        if feature_key not in crop_features:
            image = read_rgb(image_path(gallery[gi]))
            crop_features[feature_key] = selector.images([crop_person(image, c.box) for c in candidates])
        text_key = tuple(texts)
        if text_key not in text_features:
            text_features[text_key] = selector.texts(texts)
        sim = text_features[text_key] @ crop_features[feature_key].T
        members, anchors = select_subject_members(sim, candidates,
            threshold=float(selector_cfg["membership_threshold"]), margin=float(selector_cfg["membership_margin"]),
            max_members=int(selector_cfg["max_members_per_subject"]))
        result.append([[{"box": list(candidates[j].box), "global_person_index": refs[j],
                         "fallback": candidates[j].fallback, "anchor": j == anchors[si]}
                        for j in people] for si, people in enumerate(members)])
        similarities.extend(float(sim[si, j]) for si, j in enumerate(anchors))
        counts.extend(map(len, members))
        extras += sum(len(people) - 1 for people in members)
        fallback_anchors += sum(candidates[j].fallback for j in anchors)
    stats = {"selector": "openai_clip", "selector_model": "ViT-B/32",
             "assignment": "hungarian_anchors_then_threshold_expansion",
             "membership_threshold": selector_cfg["membership_threshold"],
             "membership_margin": selector_cfg["membership_margin"],
             "max_members_per_subject": selector_cfg["max_members_per_subject"],
             "membership_policy": "case_agnostic_anchor_first_disjoint_extras",
             "uses_case_labels_or_identity_counts": False,
             "predicted_member_count_histogram": dict(Counter(counts)),
             "predicted_total_query_components": sum(counts), "extra_member_components": extras,
             "selected_fallback_anchors": fallback_anchors, "full_scene_fallback_slots": padded,
             "queries_with_too_few_candidates": shortage_queries,
             "shortage_policy": "pad_missing_slots_with_full_reference_image",
             "mean_assigned_clip_similarity": float(np.mean(similarities)) if similarities else None,
             "min_assigned_clip_similarity": min(similarities) if similarities else None}
    return result, stats


def validate_localization(members, queries, gallery, detections, member_cap=5):
    if len(members) != len(queries):
        raise ValueError("Localization query count mismatch")
    index = {g["image_id"]: i for i, g in enumerate(gallery)}
    for subjects, query in zip(members, queries):
        image_id, texts = public_query(query)
        gi = index[image_id]
        if len(subjects) != len(texts):
            raise ValueError("Localization Subject count mismatch")
        used = set()
        for people in subjects:
            if not people or len(people) > member_cap or sum(p["anchor"] for p in people) != 1 or not people[0]["anchor"]:
                raise ValueError("Subject must have one leading anchor and bounded membership")
            for p in people:
                box = np.asarray(p["box"], dtype=float)
                if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    raise ValueError("Invalid localization box")
                ref = p["global_person_index"]
                if ref is None:
                    width, height = detections.image_sizes[gi]
                    if not p["fallback"] or not np.array_equal(box, [0, 0, width, height]) or not p["anchor"]:
                        raise ValueError("Invalid padded fallback")
                else:
                    if (not isinstance(ref, int) or ref < detections.offsets[gi] or ref >= detections.offsets[gi + 1]
                            or ref in used or not np.array_equal(box, detections.boxes[ref])
                            or bool(p["fallback"]) != bool(detections.fallback[ref])):
                        raise ValueError("Invalid/duplicate predicted person membership")
                    used.add(ref)


def localization_fingerprint(query_manifest, detections, cfg=None):
    cfg = cfg or protocol_config()
    selector_cfg = cfg["selector"]
    checkpoint = resolve(selector_cfg["checkpoint"])
    checksum = sha256_file(checkpoint)
    if checksum != selector_cfg["checkpoint_sha256"]:
        raise RuntimeError("Shared CLIP selector checkpoint checksum mismatch")
    return {"adapter_version": PROTOCOL_VERSION, "detections": detections.fingerprint,
                   "detection_array_sha256": detection_array_sha256(detections),
                   "queries_manifest_sha256": sha256_file(query_manifest),
                   "selector": {k: v for k, v in selector_cfg.items()
                                if k not in {"checkpoint", "checkpoint_url", "tokenizer_runtime_dir"}},
                   "selector_checkpoint_sha256": checksum, "image": cfg["image"]}


def get_localization(queries, query_manifest, gallery, detections, device, *, selector=None, cfg=None):
    cfg = cfg or protocol_config()
    selector_cfg = cfg["selector"]
    fingerprint = localization_fingerprint(query_manifest, detections, cfg)
    cache = StageCache("query_localization", fingerprint)

    def load():
        path = cache.find("members.json")
        if path is None:
            return None
        data = read_json(path)
        try:
            validate_localization(data["members"], queries, gallery, detections, int(selector_cfg["max_members_per_subject"]))
            if data["stats"]["predicted_total_query_components"] != sum(len(p) for s in data["members"] for p in s):
                raise ValueError("Localization component-count mismatch")
            return Localization(data["members"], data["stats"], fingerprint, path, True)
        except (ValueError, TypeError, KeyError):
            return None

    cached = load()
    if cached is not None:
        return cached
    if os.environ.get("CPR_REQUIRE_SHARED_CACHE") == "1":
        raise RuntimeError("No verified shared query localization; selector inference disabled by cache guard")
    with cache_lock(cache.directory):
        cached = load()
        if cached is not None:
            return cached
        if selector is None:
            if detections.gallery_manifest is None:
                raise ValueError("Selector inference requires the detection gallery manifest from get_detections")
            # Fresh imports in the worker use the fixed selector tokenizer,
            # regardless of a native model's already-imported ftfy version.
            subprocess.run([sys.executable,"-m","cpr_person.selector_worker",
                "--gallery-manifest",str(detections.gallery_manifest),
                "--query-manifest",str(resolve(query_manifest)), "--detections",str(detections.path),
                "--cache-dir",str(cache.directory),"--device",str(device)],cwd=ROOT,check=True)
            result=load()
            if result is None:
                raise RuntimeError("Shared selector worker did not produce valid localization")
            result.cache_hit=False
            return result
        members, stats = localize_queries(queries, gallery, detections, selector_cfg, selector)
        validate_localization(members, queries, gallery, detections, int(selector_cfg["max_members_per_subject"]))
        path = cache.write_json("members.json", {"members": members, "stats": stats})
        return Localization(members, stats, fingerprint, path, False)
