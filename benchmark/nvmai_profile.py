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
DEFAULT_MODEL_PATH = ROOT / "models/ornith-1.5_35B_A3B_4Bit"
DEFAULT_API_MODEL = "ornith-1.5-35b-a3b"
DEFAULT_CONTEXT_TOKENS = 262_144
DEFAULT_PROMPT_CACHE_MODE = "multi-prefix"
DEFAULT_KV_BITS = 8
DEFAULT_CONCISE = True
DEFAULT_FAST_ALIAS = False
DEFAULT_MTP = False


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
) -> list[str]:
    """Build the standard native-context, non-MTP server command."""
    return [
        str(binary),
        "--port", str(port),
        "--model", str(model),
        "--max-context", str(DEFAULT_CONTEXT_TOKENS),
        "--rope-scaling", "none",
        "--prompt-cache-mode", cache_mode,
        "--kv-bits", str(DEFAULT_KV_BITS),
    ]


def server_environment(
    base: Mapping[str, str] | None = None,
    *,
    concise: bool = DEFAULT_CONCISE,
) -> dict[str, str]:
    """Return an environment with Concise Mode explicitly selected."""
    environment = dict(os.environ if base is None else base)
    if concise:
        environment["NVMAI_CONCISE_MODE"] = "1"
    else:
        environment.pop("NVMAI_CONCISE_MODE", None)
    return environment


def request_model(*, fast: bool = DEFAULT_FAST_ALIAS) -> str:
    """Return the base API model unless an experiment explicitly asks for fast."""
    return DEFAULT_API_MODEL + ("-fast" if fast else "")
