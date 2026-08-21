import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import ornith_concise_tool_ab as benchmark


class OrnithConciseToolABTests(unittest.TestCase):
    def test_default_profile_runs_standard_before_concise(self) -> None:
        with patch.object(sys, "argv", ["ornith_concise_tool_ab.py"]):
            self.assertEqual(benchmark.parse_args().profiles, ["off", "on"])

    def test_safe_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory(dir=benchmark.ROOT / ".build") as directory:
            workspace = pathlib.Path(directory)
            with self.assertRaises(ValueError):
                benchmark.safe_path(workspace, "../outside")
            self.assertEqual(
                benchmark.safe_path(workspace, "nested/file.py"),
                workspace / "nested/file.py",
            )

    def test_execute_tool_writes_only_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir=benchmark.ROOT / ".build") as directory:
            workspace = pathlib.Path(directory)
            result = benchmark.execute_tool(
                "write_workspace_file",
                {"path": "answer.py", "content": "print('ok')\n"},
                workspace,
            )
            self.assertTrue(result["ok"])
            self.assertEqual((workspace / "answer.py").read_text(), "print('ok')\n")

    def test_self_scaffold_artifact_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=benchmark.ROOT / ".build") as directory:
            workspace = pathlib.Path(directory)
            curriculum = [{
                "language": language,
                "title": "Task",
                "objective": "Objective",
                "scaffold": f"scaffold.{suffix}",
                "self_chosen_edge_case": "Unicode",
                "rubric": ["Passes"],
            } for language, suffix in (("Python", "py"), ("Swift", "swift"))]
            (workspace / "curriculum.json").write_text(json.dumps(curriculum))
            (workspace / "reflection.json").write_text(json.dumps({
                "outcomes": ["pass", "pass"],
                "lessons": ["normalize boundaries"],
                "next_task": "Add streaming input",
            }))
            self.assertEqual(benchmark.validate_json_artifacts(workspace), [])
            (workspace / "curriculum.json").write_text(json.dumps({"tasks": curriculum}))
            self.assertEqual(benchmark.validate_json_artifacts(workspace), [])


if __name__ == "__main__":
    unittest.main()
