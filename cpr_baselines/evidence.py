"""Keep small scientific receipts portable beside completed experiment runs."""

from pathlib import Path
import shutil

from cpr_person.cache import sha256_file


def persist_evidence(run, directory, root):
    directory = Path(directory)
    target = directory / "evidence"
    target.mkdir(parents=True, exist_ok=True)

    def copy(record, name, source_key="artifact", hash_key="artifact_sha256"):
        source = Path(record[source_key])
        source = source if source.is_absolute() else root / source
        if sha256_file(source) != record[hash_key]:
            raise ValueError("Scientific receipt checksum differs")
        destination = target / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        record[source_key] = str(destination.relative_to(root))

    if run["method"] == "instruct_reid_setmatch":
        proof = run["setmatch"].get("calibration_provenance")
        if not proof:
            raise ValueError(
                "Instruct-ReID requires a frozen native-distance selection/calibration receipt"
            )
        copy(
            proof,
            (
                "unmatched_selection.json"
                if proof.get("selection_split") == "val"
                else "unmatched_external.json"
            ),
        )
    if run["method"] == "word4per_setmatch":
        proof = run.get("reproduction_provenance")
        if not proof:
            raise ValueError("Word4Per requires an admitted reproduction receipt")
        copy(proof, "stage2_reproduction.json")
    if run["method"] == "wiser":
        trace = run.get("refiner_api_trace")
        if not isinstance(trace, dict):
            raise ValueError("WISER must report its native API response trace")
        for i, record in enumerate(trace["files"]):
            copy(record, f"refiner_{i}.jsonl", "path", "sha256")
    run["selection_artifacts"] = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for name in ("tuning.json", "alpha_selection.json", "rerank_selection.json")
        if (path := directory / name).is_file()
    ]
    return run
