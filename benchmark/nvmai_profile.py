"""Shared production profile for NVMAI benchmark launchers.

Specialized A/B scripts may override the one control they measure, but every
other setting should come from this module so a plain benchmark run matches
the user-facing launcher profile.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models/ornith-1.5_35B_A3B_8Bit"
DEFAULT_API_MODEL = "ornith-1.5-35b-a3b"
DEFAULT_CONTEXT_TOKENS = 262_144
DEFAULT_PROMPT_CACHE_MODE = "multi-prefix"
DEFAULT_PROMPT_CACHE_MEMORY_MIB = 256
DEFAULT_EXPERT_CACHE_BUDGET = "8G"
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
) -> list[str]:
    """Build the standard native-context, cache-on, non-MTP command."""
    prompt_cache_memory_mib = (
        DEFAULT_PROMPT_CACHE_MEMORY_MIB if cache_mode != "off" else 0
    )
    if thinking_mode not in SUPPORTED_THINKING_MODES:
        raise ValueError("thinking_mode must be off or on")
    return [
        str(binary),
        "--port", str(port),
        "--model", str(model),
        "--max-context", str(DEFAULT_CONTEXT_TOKENS),
        "--rope-scaling", "none",
        "--prompt-cache-mode", cache_mode,
        "--prompt-cache-memory-mib", str(prompt_cache_memory_mib),
        "--ram-budget", DEFAULT_EXPERT_CACHE_BUDGET,
        "--kv-bits", str(DEFAULT_KV_BITS),
        "--thinking", thinking_mode,
    ]


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


def request_model(*, fast: bool = DEFAULT_FAST_ALIAS) -> str:
    """Return the base API model unless an experiment explicitly asks for fast."""
    return DEFAULT_API_MODEL + ("-fast" if fast else "")
