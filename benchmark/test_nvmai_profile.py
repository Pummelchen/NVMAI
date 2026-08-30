import os
import sys
import unittest
from unittest.mock import patch

import coder_cli_benchmark
import nvmai_benchmark
from nvmai_profile import (
    DEFAULT_API_MODEL,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_CONCISE,
    DEFAULT_EXPERT_CACHE_BUDGET,
    DEFAULT_KV_BITS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_CACHE_MEMORY_MIB,
    request_model,
    server_command,
    server_environment,
    configured_thinking_mode,
)


class BenchmarkProfileTests(unittest.TestCase):
    def test_server_command_matches_production_profile(self) -> None:
        command = server_command("NVMAIServer", 8081)
        self.assertEqual(DEFAULT_MODEL_PATH.name, "ornith-1.5_35B_A3B_8Bit")
        self.assertEqual(command[command.index("--model") + 1], str(DEFAULT_MODEL_PATH))
        self.assertEqual(
            command[command.index("--max-context") + 1], str(DEFAULT_CONTEXT_TOKENS)
        )
        self.assertEqual(command[command.index("--prompt-cache-mode") + 1], "multi-prefix")
        self.assertEqual(
            command[command.index("--prompt-cache-memory-mib") + 1],
            str(DEFAULT_PROMPT_CACHE_MEMORY_MIB),
        )
        self.assertEqual(
            "--ram-budget" in command, DEFAULT_EXPERT_CACHE_BUDGET is not None
        )
        self.assertEqual(command[command.index("--kv-bits") + 1], str(DEFAULT_KV_BITS))
        self.assertEqual(command[command.index("--rope-scaling") + 1], "none")
        self.assertEqual(command[command.index("--thinking") + 1], "off")
        self.assertNotIn("--mtp-model", command)

    def test_explicit_cache_off_is_not_a_default(self) -> None:
        command = server_command("NVMAIServer", 8081, cache_mode="off")
        self.assertEqual(command[command.index("--prompt-cache-mode") + 1], "off")
        self.assertEqual(command[command.index("--prompt-cache-memory-mib") + 1], "0")

    def test_shell_launchers_explicitly_enable_both_caches(self) -> None:
        launcher = (DEFAULT_MODEL_PATH.parents[1] / "tools/server_launcher.sh").read_text()
        self.assertIn("--prompt-cache-mode multi-prefix", launcher)
        self.assertIn(
            f"--prompt-cache-memory-mib {DEFAULT_PROMPT_CACHE_MEMORY_MIB}", launcher
        )
        # The expert-cache budget is per-family now (decodeTuning), so neither
        # the launcher nor the benchmark protocol pins one size: both measure
        # and ship whatever the runtime picks for the model being loaded.
        if DEFAULT_EXPERT_CACHE_BUDGET is None:
            self.assertNotIn("--ram-budget", launcher)
        else:
            self.assertIn(f"--ram-budget {DEFAULT_EXPERT_CACHE_BUDGET}", launcher)
        self.assertIn('${NVMAI_THINKING_MODE:-off}', launcher)
        self.assertIn('--thinking "$thinking_mode"', launcher)
        self.assertIn('1) quant=8bit', launcher)
        self.assertIn('case "${mode_choice:-1}"', launcher)

    def test_environment_and_model_select_standard_base_alias(self) -> None:
        environment = server_environment({"PATH": os.environ.get("PATH", "")})
        self.assertFalse(DEFAULT_CONCISE)
        self.assertNotIn("NVMAI_CONCISE_MODE", environment)
        self.assertEqual(environment["NVMAI_THINKING_MODE"], "off")
        self.assertEqual(request_model(), DEFAULT_API_MODEL)

    def test_thinking_mode_is_binary_and_configurable(self) -> None:
        self.assertEqual(configured_thinking_mode({}), "off")
        self.assertEqual(
            configured_thinking_mode({"NVMAI_THINKING_MODE": "yes"}), "on")
        with self.assertRaises(ValueError):
            configured_thinking_mode({"NVMAI_THINKING_MODE": "medium"})
        command = server_command("server", 8080, thinking_mode="on")
        self.assertEqual(command[command.index("--thinking") + 1], "on")
        self.assertFalse(request_model().endswith("-fast"))

    def test_coder_harness_plain_invocation_uses_ornith_eight_bit(self) -> None:
        with patch.object(sys, "argv", ["coder_cli_benchmark.py"]):
            arguments = coder_cli_benchmark.parse_args()
        self.assertEqual(arguments.round, "coder")
        self.assertEqual(arguments.quantizations, [8])

    def test_precise_benchmark_plain_invocation_excludes_mtp_matrix(self) -> None:
        configurations = nvmai_benchmark.selected_configs("8bit")
        self.assertEqual(
            configurations,
            [("multi-prefix", "off", "cache_on_mtp_off_8bit", 8081, 0.6)],
        )


if __name__ == "__main__":
    unittest.main()
