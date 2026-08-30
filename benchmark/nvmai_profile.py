"""Shared production profile for NVMAI benchmark launchers.

Specialized A/B scripts may override the one control they measure, but every
other setting should come from this module so a plain benchmark run matches
the user-facing launcher profile.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models/ornith-1.5_35B_A3B_8Bit"
DEFAULT_API_MODEL = "ornith-1.5-35b-a3b"
DEFAULT_CONTEXT_TOKENS = 262_144
DEFAULT_PROMPT_CACHE_MODE = "multi-prefix"
DEFAULT_PROMPT_CACHE_MEMORY_MIB = 256
# None means "do not pin a budget", so the runtime picks the family's measured
# default (RuntimeConfiguration.decodeTuning). Pinning 8G here would override the
# 12 GiB that Qwen3.8-Flash-Next now ships with, and the published protocol would
# stop measuring what a user actually gets. Pass ram_budget= explicitly to probe
# a specific size.
DEFAULT_EXPERT_CACHE_BUDGET = None
DEFAULT_KV_BITS = 8
DEFAULT_CONCISE = False
DEFAULT_FAST_ALIAS = False
DEFAULT_MTP = False
SUPPORTED_THINKING_MODES = ("off", "on")


def configured_thinking_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve the binary model switch used by every benchmark launcher."""
    source = os.environ if environment is None else environment
    value = source.get("NVMAI_THINKING_MODE", "off").lower()
    aliases = {
        "0": "off", "false": "off", "no": "off",
        "1": "on", "true": "on", "yes": "on",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_THINKING_MODES:
        raise ValueError(
            "NVMAI_THINKING_MODE must be off or on; "
            "Ornith does not expose low/medium/high effort levels"
        )
    return value


DEFAULT_THINKING_MODE = configured_thinking_mode()


def benchmark_log_path(name: str) -> str:
    """Return a git-ignored benchmark log path inside the checkout."""
    directory = ROOT / ".build/benchmark-logs"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / name)


def server_command(
    binary: str | os.PathLike[str],
    port: int,
    *,
    model: str | os.PathLike[str] = DEFAULT_MODEL_PATH,
    cache_mode: str = DEFAULT_PROMPT_CACHE_MODE,
    thinking_mode: str = DEFAULT_THINKING_MODE,
    ram_budget: str | None = DEFAULT_EXPERT_CACHE_BUDGET,
) -> list[str]:
    """Build the standard native-context, cache-on, non-MTP command."""
    prompt_cache_memory_mib = (
        DEFAULT_PROMPT_CACHE_MEMORY_MIB if cache_mode != "off" else 0
    )
    if thinking_mode not in SUPPORTED_THINKING_MODES:
        raise ValueError("thinking_mode must be off or on")
    command = [
        str(binary),
        "--port", str(port),
        "--model", str(model),
        "--max-context", str(DEFAULT_CONTEXT_TOKENS),
        "--rope-scaling", "none",
        "--prompt-cache-mode", cache_mode,
        "--prompt-cache-memory-mib", str(prompt_cache_memory_mib),
        "--kv-bits", str(DEFAULT_KV_BITS),
        "--thinking", thinking_mode,
    ]
    if ram_budget is not None:
        command += ["--ram-budget", str(ram_budget)]
    return command


def server_environment(
    base: Mapping[str, str] | None = None,
    *,
    concise: bool = DEFAULT_CONCISE,
    thinking_mode: str = DEFAULT_THINKING_MODE,
) -> dict[str, str]:
    """Return an environment with concise and thinking modes selected."""
    if thinking_mode not in SUPPORTED_THINKING_MODES:
        raise ValueError("thinking_mode must be off or on")
    environment = dict(os.environ if base is None else base)
    if concise:
        environment["NVMAI_CONCISE_MODE"] = "1"
    else:
        environment.pop("NVMAI_CONCISE_MODE", None)
    environment["NVMAI_THINKING_MODE"] = thinking_mode
    return environment


def resolve_api_model(port, *, timeout=5):
    """Ask the server which model it serves.

    The id names the quantization now (`ornith-1.5-35b-a3b_4-Bit`), and it
    differs per model, so a hardcoded default silently restricts every harness
    to one install -- which is what it did.
    """
    import http.client
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/v1/models")
        data = json.loads(conn.getresponse().read().decode())
        conn.close()
        ids = [row["id"] for row in data.get("data", [])
               if not row["id"].endswith("-fast")]
        if ids:
            return ids[0]
    except (OSError, ValueError, KeyError):
        pass
    return DEFAULT_API_MODEL


def request_model(*, fast: bool = DEFAULT_FAST_ALIAS,
                  base: str | None = None) -> str:
    """Return the base API model unless an experiment explicitly asks for fast."""
    return (base or DEFAULT_API_MODEL) + ("-fast" if fast else "")
