#!/usr/bin/env bash
# Regenerates the figures, inlines them into the template and prints to PDF.
set -euo pipefail
cd "$(dirname "$0")"
python3 figures.py >/dev/null
python3 - <<'PY'
import re, pathlib
t = pathlib.Path("paper.template.html").read_text()
t = re.sub(r"\{\{FIG:(\w+)\}\}", lambda m: pathlib.Path(f"fig/{m.group(1)}.svg").read_text(), t)
pathlib.Path("paper.html").write_text(t)
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/continuitycore-paper.pdf" "file://$PWD/paper.html" 2>/dev/null
python3 -c "
import re,sys; d=open('continuitycore-paper.pdf','rb').read(); print('pages:', len(re.findall(rb'/Type\s*/Page[^s]', d)), 'bytes:', len(d))"
