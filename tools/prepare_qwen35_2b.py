#!/usr/bin/env python3.13
"""Qwen3.5-2B's converter, now `prepare_qwen35.py --size 2b`.

Kept so the commands recorded in docs and receipts still run unchanged, and
so anything that imported this file still finds `quantize_affine` and the
naming rules here. The converter itself lives in one place.

    python3.13 tools/prepare_qwen35_2b.py --bits 8 \\
        --output .build/qwen35-2b-affine-8bit --work .build/qwen35-2b-shards
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_qwen35 import *  # noqa: E402,F401,F403
from prepare_qwen35 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--size", "2b", *sys.argv[1:]]))
