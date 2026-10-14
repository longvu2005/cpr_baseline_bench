"""Offline tests for the Kaggle Python 3.13 ensurepip-safe environment manager."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_environment import (
    MARKER_NAME, VENV_CREATED_NAME, _child_env, _create_venv, _method_env,
    _pip_command, _requirements_digest, _venv_python_works, prepare_method_python,
)


class EnvironmentTests(unittest.TestCase):
    def test_nested_requirements_invalidate_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested.txt").write_text("numpy>=1.24,<2\n", encoding="utf-8")
            req = root / "requirements.txt"
            req.write_text("-r nested.txt\n", encoding="utf-8")
            first, contents = _requirements_digest(req)
            self.assertIn("numpy>=1.24,<2", contents)
            (root / "nested.txt").write_text("numpy>=2\n", encoding="utf-8")
            second, _ = _requirements_digest(req)
            self.assertNotEqual(first, second)

    def test_skip_install_rejects_interrupted_ensurepip_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req = root / "requirements.txt"
            req.write_text("\n", encoding="utf-8")
            directory, executable = _method_env(root, "imagine_seek")
            executable.parent.mkdir(parents=True)
            executable.touch()  # Python exists but venv bootstrap never completed.
            (directory / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
            with self.assertRaisesRegex(RuntimeError, "Missing/incomplete isolated environment"):
                prepare_method_python(root=root, method_id="imagine_seek",
                                      requirements_path=req, skip_install=True)

    def test_creates_real_venv_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory, executable = _method_env(root, "imagine_seek")
            env = _child_env(directory)
            _create_venv(directory, executable, env)
            self.assertTrue(_venv_python_works(directory, executable, env))
            self.assertTrue((directory / VENV_CREATED_NAME).is_file())
            self.assertIsInstance(_pip_command(executable, env), list)

    def test_pip_fallback_targets_venv_when_inner_pip_absent(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "bin" / "python"
            responses = iter([SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)])
            with patch("baseline_environment.subprocess.run", side_effect=lambda *a, **k: next(responses)) as run:
                cmd = _pip_command(target, {})
            self.assertEqual(cmd[:3], [sys.executable, "-m", "pip"])
            self.assertEqual(cmd[3:5], ["--python", str(target)])
            self.assertEqual(run.call_count, 2)

    def test_interrupted_venv_repaired_and_marker_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req = root / "requirements.txt"
            req.write_text("\n", encoding="utf-8")
            directory, executable = _method_env(root, "imagine_seek")
            directory.mkdir(parents=True)
            (directory / "stale-file").write_text("partial")
            # Install empty requirements (no network) and fake model-only imports.
            with patch("baseline_environment._smoke_test") as smoke:
                py, env = prepare_method_python(
                    root=root, method_id="imagine_seek", requirements_path=req, skip_install=False,
                )
                self.assertEqual(py, str(executable))
                self.assertTrue((directory / MARKER_NAME).is_file())
                self.assertFalse((directory / "stale-file").exists())
                before = json.loads((directory / MARKER_NAME).read_text())
                self.assertEqual(before["bootstrap_schema"], 2)
                with patch("baseline_environment._create_venv") as recreated:
                    prepare_method_python(root=root, method_id="imagine_seek",
                                          requirements_path=req, skip_install=True)
                    recreated.assert_not_called()
                smoke.assert_called()
            req.write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing/stale requirements"):
                prepare_method_python(root=root, method_id="imagine_seek",
                                      requirements_path=req, skip_install=True)


if __name__ == "__main__":
    unittest.main()
