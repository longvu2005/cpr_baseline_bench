"""Offline tests for reproducible, isolated method environments."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# When copied to <repo>/tests/ and run via unittest discover.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_environment import _requirements_digest, prepare_method_python


class EnvironmentTests(unittest.TestCase):
    def test_nested_requirements_invalidate_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested.txt").write_text("numpy>=1.24,<2\n")
            req = root / "requirements.txt"
            req.write_text("-r nested.txt\ngit+https://github.com/openai/CLIP.git@abc\n")
            first, contents = _requirements_digest(req)
            self.assertIn("openai/CLIP", contents)
            (root / "nested.txt").write_text("numpy>=2\n")
            second, _ = _requirements_digest(req)
            self.assertNotEqual(first, second)

    def test_skip_install_fails_before_cloned_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req = root / "requirements.txt"
            req.write_text("numpy>=1.24,<2\n")
            with self.assertRaisesRegex(RuntimeError, "without --skip-install"):
                prepare_method_python(root=root, method_id="test_method",
                                      requirements_path=req, skip_install=True)

    def test_prepared_env_is_reused_but_stale_env_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req = root / "requirements.txt"
            req.write_text("numpy>=1.24,<2\n")
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if command[1:3] == ["-m", "venv"]:
                    (root / ".baseline_venvs" / "test_method" / "bin").mkdir(parents=True)
                    (root / ".baseline_venvs" / "test_method" / "bin" / "python").touch()
                return None

            with patch("baseline_environment.subprocess.run", side_effect=fake_run):
                python1, env1 = prepare_method_python(root=root, method_id="test_method",
                                                      requirements_path=req, skip_install=False)
                python2, env2 = prepare_method_python(root=root, method_id="test_method",
                                                      requirements_path=req, skip_install=True)
            self.assertEqual(python1, python2)
            self.assertEqual(env1["USE_FLAX"], "0")
            self.assertEqual(env2["USE_TF"], "0")
            self.assertEqual(sum("pip" in cmd for cmd in calls), 1)
            req.write_text("numpy>=1.26,<2\n")
            with self.assertRaisesRegex(RuntimeError, "missing/stale requirements"):
                prepare_method_python(root=root, method_id="test_method",
                                      requirements_path=req, skip_install=True)


if __name__ == '__main__':
    unittest.main()
