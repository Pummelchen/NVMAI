import pathlib
import tempfile
import unittest

import ornith_four_program_matrix as benchmark


class OrnithFourProgramMatrixTests(unittest.TestCase):
    def test_matrix_has_exactly_eight_unique_cells(self) -> None:
        cells = {
            (quant, concise, thinking)
            for quant in (4, 8)
            for concise in (False, True)
            for thinking in (False, True)
        }
        self.assertEqual(len(cells), 8)
        self.assertEqual(benchmark.MAX_TURNS_PER_PROGRAM, 25)

    def test_safe_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory(dir=benchmark.ROOT / ".build") as directory:
            workspace = pathlib.Path(directory)
            with self.assertRaises(ValueError):
                benchmark.safe_path(workspace, "../outside.py")

    def test_diagnostic_artifacts_are_readable_but_not_directly_writable(self) -> None:
        with tempfile.TemporaryDirectory(dir=benchmark.ROOT / ".build") as directory:
            workspace = pathlib.Path(directory)
            (workspace / "diagnostic.txt").write_text("shape=3x3\n")
            read = benchmark.execute_tool(
                "read_workspace_file", {"path": "diagnostic.txt"}, workspace)
            write = benchmark.execute_tool(
                "write_workspace_file",
                {"path": "diagnostic.txt", "content": "forged\n"},
                workspace,
            )
            self.assertTrue(read["ok"])
            self.assertFalse(write["ok"])

    def test_expected_hidden_outputs_form_the_final_result(self) -> None:
        combined = " | ".join(
            benchmark.EXPECTED_HIDDEN[name].strip()
            for name in benchmark.PROGRAM_ORDER
        )
        self.assertEqual(combined, benchmark.EXPECTED_FINAL)

    def test_runtime_routing_uses_the_required_environments(self) -> None:
        workspace = benchmark.ROOT / ".build"
        swift, _ = benchmark.runtime_command(workspace / "ShortestPath.swift")
        python, _ = benchmark.runtime_command(workspace / "NestedScore.py")
        tensorflow, _ = benchmark.runtime_command(workspace / "StackedLSTM.py")
        pytorch, _ = benchmark.runtime_command(workspace / "MaskedAttention.py")
        self.assertEqual(swift[:2], ["xcrun", "swift"])
        self.assertEqual(pathlib.Path(python[0]), benchmark.PYTHON_314)
        self.assertEqual(pathlib.Path(tensorflow[0]), benchmark.TENSORFLOW_PYTHON)
        self.assertEqual(pathlib.Path(pytorch[0]), benchmark.PYTORCH_PYTHON)

    def test_each_conversation_is_scoped_to_one_assigned_program(self) -> None:
        self.assertIn("one assigned file", benchmark.SYSTEM_PROMPT)
        self.assertNotIn("all four public checks", benchmark.SYSTEM_PROMPT)

    def test_matrix_table_reports_full_wall_and_model_wait(self) -> None:
        record = {
            "quant_bits": 8,
            "concise": False,
            "thinking": False,
            "wall_seconds": 12.5,
            "model_wall_seconds": 10.25,
            "usage_totals": {"completion_tokens": 42},
            "quality": {"passed": True, "final_result": benchmark.EXPECTED_FINAL},
            "error": None,
        }
        table = benchmark.matrix_table([record])
        self.assertIn("| Wall | Model wait |", table)
        self.assertIn("| 12.5s | 10.2s | 42 |", table)


if __name__ == "__main__":
    unittest.main()
