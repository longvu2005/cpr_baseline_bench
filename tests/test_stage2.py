"""Stage-2 ingestion and adapter checks; no model weights or GPU required."""

import ast
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import build_tables
import evaluate
import run_baseline
from benchmark_data import manifest_fingerprint
from link_gallery import link_gallery
from prepare_data import CASE_TYPES, build_queries, query_texts, read_jsonl, select_gallery_images, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def gallery_fixture():
    return [{"gallery_idx": index, "image_id": image_id,
             "path": f"data/gallery/{image_id}.jpg", "person_ids": people}
            for index, (image_id, people) in enumerate([
                ("q", ["10", "20", "30"]), ("t", ["10", "20"]),
                ("p", ["10", "20", "30"]), ("d", ["99"]),
            ])]


def record_fixture():
    desc = "Identify Subject 1 as the man wearing a white shirt"
    change = "then retrieve target images where Subject 1 is sitting"
    return {"sample_id": "train__q__t", "case_type": "INDIVIDUAL",
            "query_image_id": "q", "target_image_id": "t",
            "positive_image_ids": ["t", "p"],
            "subjects": [{"subject_id": 1, "identity_ids": ["10"]}],
            "final_desc": desc, "final_change": change,
            "final_instruction": f"{desc}; {change}."}


def adapter_helper(relative_path, function_name, with_target=False):
    """Execute an adapter's pure text helper without importing model libraries."""
    module = ast.parse((ROOT / relative_path).read_text())
    names = {function_name, "QueryTarget"} if with_target else {function_name}
    body = [node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"Any": Any, "dataclass": dataclass, "__name__": __name__}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(relative_path), "exec"), namespace)
    return namespace[function_name]


class Stage2Tests(unittest.TestCase):
    def test_complete_positive_list_and_original_instruction(self):
        record = record_fixture()
        queries, skipped = build_queries([record], gallery_fixture())
        query = queries[0]
        self.assertEqual(query["full_positive_ids"], ["t", "p"])
        self.assertEqual(query["text"], record["final_instruction"])
        self.assertEqual(query["subjects"][0]["select_text"], "the man wearing a white shirt")
        self.assertEqual(query["subjects"][0]["modify_text"], "is sitting")
        self.assertFalse(skipped)

    def test_independent_dual_clauses(self):
        record = record_fixture()
        record["final_desc"] += " and Subject 2 as the woman wearing glasses"
        record["final_change"] = (
            "then retrieve target images where Subject 1 is sitting and Subject 2 is standing"
        )
        _, selectors, modifiers, relation = query_texts(record)
        self.assertEqual(selectors, {1: "the man wearing a white shirt", 2: "the woman wearing glasses"})
        self.assertEqual(modifiers, {1: "is sitting", 2: "is standing"})
        self.assertIsNone(relation)

    def test_reversed_condition_order(self):
        record = record_fixture()
        record["final_desc"] += " and Subject 2 as the woman wearing glasses"
        record["final_change"] = (
            "then retrieve target images where Subject 2 is standing, while Subject 1 is sitting"
        )
        self.assertEqual(query_texts(record)[2], {2: "is standing", 1: "is sitting"})

    def test_relational_and_shared_conditions_are_preserved(self):
        record = record_fixture()
        record["final_desc"] += ", and Subject 2 as the woman wearing glasses"
        conditions = [
            "Subject 1 is standing behind Subject 2",
            "Subject 1 is smiling, and Subject 2 is holding Subject 1's hand",
            "Subject 1 and Subject 2 are sitting together",
            "Subject 2 is standing beside Subject 1, while Subject 1 is smiling",
        ]
        for condition in conditions:
            with self.subTest(condition=condition):
                record["final_change"] = "then retrieve target images where " + condition
                _, _, modifiers, relation = query_texts(record)
                self.assertEqual(relation, condition)
                self.assertEqual(modifiers, {1: condition, 2: condition})

    def test_group_has_one_text_slot_and_all_identity_labels(self):
        record = record_fixture()
        record["case_type"] = "GROUP"
        record["subjects"][0]["identity_ids"] = ["10", "20"]
        record["final_desc"] = "Identify Subject 1 as the couple wearing white shirts"
        record["final_change"] = "then retrieve target images where Subject 1 are sitting"
        query = build_queries([record], gallery_fixture())[0][0]
        self.assertEqual(query["target_ids"], ["10", "20"])
        self.assertEqual(len(query["subjects"]), 1)
        self.assertEqual(query["subjects"][0]["identity_ids"], ["10", "20"])
        self.assertEqual(query["subjects"][0]["modify_text"], "are sitting")

    def test_case_and_gt_counts_do_not_route_adapter_inputs(self):
        record = record_fixture()
        original = build_queries([record], gallery_fixture())[0][0]
        record["case_type"] = "RELATIONAL"
        record["subjects"][0]["identity_ids"] = ["10", "20"]
        changed = build_queries([record], gallery_fixture())[0][0]
        for field in ("text", "relation_text"):
            self.assertEqual(original[field], changed[field])
        self.assertEqual(len(original["subjects"]), len(changed["subjects"]))
        for field in ("select_text", "modify_text"):
            self.assertEqual(original["subjects"][0][field], changed["subjects"][0][field])

    def test_positive_filtering_is_stable_and_reported(self):
        record = record_fixture()
        record["positive_image_ids"] = ["p", "q", "outside", "t", "p"]
        query, skipped = build_queries([record], gallery_fixture())
        self.assertEqual(query[0]["full_positive_ids"], ["p", "t"])
        self.assertEqual(dict(skipped), {"self_positive_removed": 1, "positive_outside_gallery": 1})

    def test_queries_outside_the_selected_gallery_are_excluded(self):
        records = [record_fixture()]
        for split in ("val", "test"):
            record = record_fixture()
            record.update(sample_id=f"{split}__outside__target", query_image_id="outside")
            records.append(record)
        queries, skipped = build_queries(records, gallery_fixture())
        self.assertEqual(len(queries), 1)
        self.assertEqual(skipped["query_outside_gallery"], 2)

    def test_order_does_not_depend_on_input_row_order(self):
        a = record_fixture()
        b = copy.deepcopy(a)
        b["sample_id"] = "train__q__a"
        expected = build_queries([a, b], gallery_fixture())[0]
        self.assertEqual(expected, build_queries([b, a], gallery_fixture())[0])
        self.assertEqual([q["query_idx"] for q in expected], [0, 1])
        self.assertEqual(expected[0]["query_id"], "train__q__a")

    def test_malformed_labels_and_text_fail(self):
        for mutation, message in [
            (lambda r: r.update(final_desc=""), "final_desc"),
            (lambda r: r.update(positive_image_ids=["p"]), "non-self positive"),
            (lambda r: r.update(positive_image_ids=["t", "d"]), "all target identities"),
            (lambda r: r.update(final_change="Subject 2 is standing"), "undefined Subject"),
            (lambda r: r["subjects"][0].update(subject_id=2), "Subject names"),
        ]:
            with self.subTest(message=message):
                record = record_fixture()
                mutation(record)
                with self.assertRaisesRegex(ValueError, message):
                    build_queries([record], gallery_fixture())
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            build_queries([record_fixture(), record_fixture()], gallery_fixture())
        with self.assertRaisesRegex(ValueError, "flattened export"):
            build_queries([{"submission_id": "old"}], gallery_fixture())

    def test_jsonl_bom_blank_lines_and_object_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.txt"
            path.write_text("\ufeff" + json.dumps(record_fixture()) + "\n\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [record_fixture()])
            path.write_text("[]\n")
            with self.assertRaisesRegex(TypeError, "JSONL row must be an object"):
                read_jsonl(path)

    def test_person_adapter_parsers_accept_group_and_shared_text(self):
        record = record_fixture()
        record["case_type"] = "GROUP"
        record["subjects"][0]["identity_ids"] = ["10", "20"]
        query = build_queries([record], gallery_fixture())[0][0]
        for directory in ("01_word4per_setmatch", "02_fafa_setmatch", "08_basic_setmatch"):
            parse = adapter_helper(f"methods/published/{directory}/run.py", "parse_query_targets", True)
            target = parse(query, 0)
            self.assertEqual(len(target), 1)
            self.assertEqual(target[0].modify_text, "is sitting")
            self.assertEqual(target[0].select_text, "the man wearing a white shirt")

    def test_composition_helpers_use_text_instead_of_case(self):
        query = {"case": "GROUP", "text": "complete instruction",
                 "relation_text": "Subject 1 holds Subject 2's hand"}
        subject = {"modify_text": "short modifier"}
        for relative in (
            "methods/simple/10_person_compose_setmatch/run.py",
            "methods/published/09_adafocal_setmatch/run.py",
        ):
            function = adapter_helper(relative, "query_compose_text")
            self.assertEqual(function(query, subject), "complete instruction")
            query["case"] = "RELATIONAL"
            query["relation_text"] = None
            self.assertEqual(function(query, subject), "short modifier")
            query.update(case="GROUP", relation_text="Subject 1 holds Subject 2's hand")


class GalleryTests(unittest.TestCase):
    def test_metadata_gallery_includes_all_three_splits_in_image_order(self):
        images = [{"image_id": split.lower(), "source_split": split, "image_idx": index}
                  for split, index in (("TEST", 3), ("TRAIN", 1), ("OTHER", 0), ("VAL", 2))]
        self.assertEqual([row["image_id"] for row in select_gallery_images(images)],
                         ["train", "val", "test"])

    def test_flat_links_replace_old_train_folder_link_and_can_be_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images"
            for split in ("train", "val", "test"):
                folder = source / split
                folder.mkdir(parents=True)
                (folder / f"{split}.jpg").write_bytes(split.encode())
            gallery = root / "gallery"
            gallery.symlink_to(source / "train", target_is_directory=True)
            self.assertEqual(dict(link_gallery(source, gallery)), {"train": 1, "val": 1, "test": 1})
            self.assertFalse(gallery.is_symlink())
            for split in ("train", "val", "test"):
                image = gallery / f"{split}.jpg"
                self.assertTrue(image.is_symlink())
                self.assertEqual(image.read_bytes(), split.encode())
                self.assertEqual((source / split / f"{split}.jpg").read_bytes(), split.encode())
            link_gallery(source, gallery)
            self.assertEqual(len(list(gallery.iterdir())), 3)

    def test_source_collisions_leave_the_original_gallery_link_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images"
            for split in ("train", "val", "test"):
                folder = source / split
                folder.mkdir(parents=True)
                (folder / "same.jpg").write_bytes(split.encode())
            gallery = root / "gallery"
            gallery.symlink_to(source / "train", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Duplicate image filename"):
                link_gallery(source, gallery)
            self.assertTrue(gallery.is_symlink())
            self.assertEqual((gallery / "same.jpg").read_bytes(), b"train")

    def test_existing_real_images_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images"
            for split in ("train", "val", "test"):
                folder = source / split
                folder.mkdir(parents=True)
                (folder / f"{split}.jpg").write_bytes(split.encode())
            gallery = root / "gallery"
            gallery.mkdir()
            (gallery / "train.jpg").write_bytes(b"existing image")
            with self.assertRaisesRegex(FileExistsError, "non-symlink image"):
                link_gallery(source, gallery)
            self.assertEqual((gallery / "train.jpg").read_bytes(), b"existing image")


class EvaluationTests(unittest.TestCase):
    def test_multiple_positives_group_metrics_and_current_data_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            run_dir = root / "runs" / "fixture"
            run_dir.mkdir(parents=True)
            records = [record_fixture()]
            group = record_fixture()
            group.update(sample_id="train__q__group", case_type="GROUP")
            group["subjects"][0]["identity_ids"] = ["10", "20"]
            records.append(group)
            gallery = gallery_fixture()
            queries = build_queries(records, gallery)[0]
            write_jsonl(root / "data/gallery.jsonl", gallery)
            write_jsonl(root / "data/queries.jsonl", queries)
            fingerprint = manifest_fingerprint(root)
            # Self is highest; after exclusion positives have ranks 1 and 3.
            np.save(run_dir / "scores.npy", np.tile([9.0, 3.0, 1.0, 2.0], (2, 1)))
            run = {"method": "fixture", "higher_is_better": True,
                   "data_fingerprint": fingerprint}
            (run_dir / "run.json").write_text(json.dumps(run))
            overrides = {"ROOT": root, "DATA_DIR": root / "data",
                         "RUNS_DIR": root / "runs", "OUTPUTS_DIR": root / "outputs"}
            with patch.multiple(evaluate, **overrides), patch.object(evaluate, "progress_bar", lambda iterable, **kwargs: iterable), patch("sys.argv", ["evaluate.py", "--method", "fixture"]), redirect_stdout(io.StringIO()):
                evaluate.main()
            metrics = json.loads((root / "outputs/fixture/metrics.json").read_text())
            self.assertAlmostEqual(metrics["overall"]["Full-mAP"], 5 / 6)
            self.assertAlmostEqual(metrics["cases"]["GROUP"]["Full-mAP"], 5 / 6)
            self.assertEqual(metrics["data_fingerprint"], fingerprint)

            old_dir = root / "outputs/previous"
            old_dir.mkdir()
            (old_dir / "metrics.json").write_text(json.dumps({"method": "previous"}))
            with patch.multiple(build_tables, ROOT=root, OUTPUTS_DIR=root / "outputs"), redirect_stdout(io.StringIO()):
                results = build_tables.collect_results()
            self.assertEqual([result["method_id"] for result in results], ["fixture"])
            table = build_tables.build_case_table(results)[0]
            self.assertEqual(table["GROUP mAP"], "83.33")
            for case in CASE_TYPES:
                self.assertIn(f"{case} mAP", build_tables.CASE_COLUMNS)

            # A same-sized but edited manifest must invalidate the score run.
            queries[0]["text"] += " edited"
            write_jsonl(root / "data/queries.jsonl", queries)
            with patch.multiple(evaluate, **overrides), patch("sys.argv", ["evaluate.py", "--method", "fixture"]), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "manifests have changed"):
                    evaluate.main()
            with patch.multiple(build_tables, ROOT=root, OUTPUTS_DIR=root / "outputs"), redirect_stdout(io.StringIO()):
                self.assertEqual(build_tables.collect_results(), [])


class RunnerTests(unittest.TestCase):
    def run_fixture(self, root, change_manifest=False):
        (root / "data").mkdir()
        write_jsonl(root / "data/gallery.jsonl", gallery_fixture())
        queries = build_queries([record_fixture()], gallery_fixture())[0]
        write_jsonl(root / "data/queries.jsonl", queries)
        method_dir = root / "methods/simple/fixture"
        method_dir.mkdir(parents=True)
        spec = run_baseline.MethodSpec("fixture", method_dir)
        steps = []

        def execute_step(index, total, title, command):
            steps.append(index)
            if index == 4:
                directory = root / "runs/fixture"
                directory.mkdir(parents=True)
                (directory / "run.json").write_text(json.dumps({"method": "fixture"}))
                if change_manifest:
                    queries[0]["text"] += " edited during inference"
                    write_jsonl(root / "data/queries.jsonl", queries)
            if index == 5:
                run = json.loads((root / "runs/fixture/run.json").read_text())
                self.assertEqual(run["data_fingerprint"], manifest_fingerprint(root))

        with patch.object(run_baseline, "ROOT", root), patch.object(run_baseline, "ensure_gallery_layout", return_value=root / "data/gallery"), patch.object(run_baseline, "describe_gallery_link", return_value="fixture"), patch.object(run_baseline.subprocess, "run"), patch.object(run_baseline, "run_step", side_effect=execute_step), redirect_stdout(io.StringIO()):
            run_baseline.run_pipeline(spec, force_checkpoint=False, skip_install=True)
        return steps

    def test_runner_records_manifest_hashes_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_fixture(Path(directory)), [4, 5, 6])

    def test_runner_stops_when_data_changes_during_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Manifests changed during inference"):
                self.run_fixture(Path(directory), change_manifest=True)


if __name__ == "__main__":
    unittest.main()
