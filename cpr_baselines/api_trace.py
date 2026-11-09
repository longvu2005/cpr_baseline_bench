"""Preserve native API requests and record exact responses/model identities."""

from __future__ import annotations

from cpr_person.cache import atomic_json, canonical_hash, read_json


def bind_model(path, requested, resolved):
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("API response did not identify its resolved model")
    current = read_json(path)
    identity = {"requested_model": requested, "resolved_model": resolved}
    if current is not None and current != identity:
        raise RuntimeError(
            "API model identity changed within this run; keep response cohorts separate"
        )
    atomic_json(path, identity)


def complete(client, model, messages, identity_path):
    request = {"model": model, "messages": messages}
    response = client.chat.completions.create(**request)
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict) or not raw.get("id") or not raw.get("choices"):
        raise ValueError("Cannot record native API response provenance")
    content = response.choices[0].message.content or ""
    receipt = {
        "schema": "native-api-response-v1",
        "request": request,
        "request_sha256": canonical_hash(request),
        "response": raw,
        "response_sha256": canonical_hash(raw),
    }
    bind_model(identity_path, model, raw.get("model"))
    return content, receipt


def validate_cached(receipt, model, messages, content, identity_path):
    request = {"model": model, "messages": messages}
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "native-api-response-v1"
        or receipt.get("request") != request
        or receipt.get("request_sha256") != canonical_hash(request)
    ):
        raise ValueError("Cached native API request differs")
    raw = receipt.get("response")
    if not isinstance(raw, dict) or receipt.get("response_sha256") != canonical_hash(
        raw
    ):
        raise ValueError("Cached native API response changed")
    if (
        not raw.get("id")
        or not raw.get("choices")
        or (raw["choices"][0]["message"].get("content") or "") != content
    ):
        raise ValueError("Cached native API content differs from response")
    bind_model(identity_path, model, raw.get("model"))


def summarize(directory, root):
    import json
    from pathlib import Path
    from cpr_person.cache import sha256_file

    files, requested, resolved = [], set(), set()
    count = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for branch, receipt in row.get("api_receipts", {}).items():
                if receipt is None:
                    continue
                request, response = receipt["request"], receipt["response"]
                if (
                    receipt.get("request_sha256") != canonical_hash(request)
                    or receipt.get("response_sha256") != canonical_hash(response)
                    or (response["choices"][0]["message"].get("content") or "")
                    != row[branch]
                ):
                    raise ValueError("Native API trace changed before completion")
                requested.add(request["model"])
                resolved.add(response["model"])
                count += 1
        files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    if len(requested) > 1 or len(resolved) > 1:
        raise RuntimeError("Mixed API model identities in native response trace")
    return {
        "schema": "wiser-native-api-trace-v1",
        "num_calls": count,
        "requested_models": sorted(requested),
        "resolved_models": sorted(resolved),
        "files": files,
    }
