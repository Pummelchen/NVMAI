import os
import sys
import unittest
from unittest.mock import patch

import coder_cli_benchmark
import nvmai_benchmark
from nvmai_profile import (
    DEFAULT_API_MODEL,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_KV_BITS,
    DEFAULT_MODEL_PATH,
    request_model,
    server_command,
    server_environment,
)


class BenchmarkProfileTests(unittest.TestCase):
    def test_server_command_matches_production_profile(self) -> None:
        command = server_command("NVMAIServer", 8081)
        self.assertEqual(command[command.index("--model") + 1], str(DEFAULT_MODEL_PATH))
        self.assertEqual(
            command[command.index("--max-context") + 1], str(DEFAULT_CONTEXT_TOKENS)
        )
        self.assertEqual(command[command.index("--prompt-cache-mode") + 1], "multi-prefix")
        self.assertEqual(command[command.index("--kv-bits") + 1], str(DEFAULT_KV_BITS))
        self.assertEqual(command[command.index("--rope-scaling") + 1], "none")
        self.assertNotIn("--mtp-model", command)

    def test_environment_and_model_select_concise_base_alias(self) -> None:
        environment = server_environment({"PATH": os.environ.get("PATH", "")})
        self.assertEqual(environment["NVMAI_CONCISE_MODE"], "1")
        self.assertEqual(request_model(), DEFAULT_API_MODEL)
        self.assertFalse(request_model().endswith("-fast"))

    def test_coder_harness_plain_invocation_uses_ornith_four_bit(self) -> None:
        with patch.object(sys, "argv", ["coder_cli_benchmark.py"]):
            arguments = coder_cli_benchmark.parse_args()
        self.assertEqual(arguments.round, "coder")
        self.assertEqual(arguments.quantizations, [4])

    def test_precise_benchmark_plain_invocation_excludes_mtp_matrix(self) -> None:
        configurations = nvmai_benchmark.selected_configs("4bit")
        self.assertEqual(
            configurations,
            [("multi-prefix", "off", "cache_on_mtp_off_4bit", 8081, 0.6)],
        )


if __name__ == "__main__":
    unittest.main()
