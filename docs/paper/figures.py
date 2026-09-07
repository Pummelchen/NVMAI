"""Generates the paper's data figures as SVG from the benchmark result
directories, so every number in a chart comes from the same files as the
tables. No plotting dependency: the SVG is written directly."""
import importlib.util, json, os, re, sys
from pathlib import Path
ROOT = Path("/Users/andreborchert/Downloads/NVMAI"); LOGS = ROOT / ".build/benchmark-logs"
OUT = ROOT / "docs/paper/fig"; OUT.mkdir(exist_ok=True)

def load_mod(name, results):
    os.environ["NVMAI_MEMVAL_RESULTS"] = str(results)
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmark" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
book = load_mod("memory_book", LOGS / "memory-book-v2")
pong = load_mod("memory_value", LOGS / "memory-value-v2")

def consolidation(d, arm, run):
    log = d / f"server-{arm}-r{run}.log"
    if not log.exists(): return 0, 0
    pairs = re.findall(r"memory consolidated .*?prompt=(\d+) completion=(\d+)", log.read_text())
    return sum(int(a) for a, _ in pairs), sum(int(b) for _, b in pairs)

def book_runs(label, arm):
    d = LOGS / f"memory-book-{label}"; rows = []
    for run in (1, 2, 3):
        p = d / f"{arm}-r{run}.json"
        if not p.exists(): continue
        res = json.loads(p.read_text()); per = {}
        c = t = req_p = req_c = sec = 0
        for r in res:
            cc, tt = book.score(r["session"], r["answers"]); per[r["session"]] = (cc, tt)
            if r["session"] > 1: c += cc; t += tt
            req_p += r["prompt_tokens"] + r.get("summary_prompt_tokens", 0)
            req_c += r["completion_tokens"] + r.get("summary_completion_tokens", 0)
            sec += r["seconds"] + r.get("summary_seconds", 0) + r.get("consolidation_wait", 0)
        cp, cc2 = consolidation(d, arm, run)
        rows.append(dict(c=c, t=t, req_p=req_p, req_c=req_c, con_p=cp, con_c=cc2, sec=sec, per=per))
    return rows

def pong_runs(label, arm):
    d = LOGS / f"memory-value-{label}"; rows = []
    for run in (1, 2, 3):
        p = d / f"{arm}-r{run}.json"
        if not p.exists(): continue
        base = {}; a = s = 0
        for r in json.loads(p.read_text()):
            v = pong.extract(r["content"])
            if r["stage"] == "swift": base = v
            else:
                sh = set(v) & set(base); s += len(sh); a += sum(1 for k in sh if v[k] == base[k])
        rows.append((a, s))
    return rows

def mean_pct(rows, key_c="c", key_t="t"):
    c = sum(r[key_c] for r in rows); t = sum(r[key_t] for r in rows)
    return 100 * c / t if t else None

# ---------- SVG helpers ----------
FONT = "font-family='Helvetica Neue, Helvetica, Arial, sans-serif'"
def svg_open(w, h): return [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT} font-size='11'>"]
def text(x, y, s, size=11, anchor="start", weight="normal", fill="#222", rot=None):
    t = f" transform='rotate({rot} {x} {y})'" if rot else ""
    return f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' fill='{fill}'{t}>{s}</text>"
PAL = ["#5b6b7f", "#2f7fb3", "#d98f2b", "#5aa469"]

def grouped_bars(path, groups, series, ymax, ylabel, w=520, h=260, fmt="{:.0f}", band=None, colors=PAL, legend=True):
    """groups: list of labels; series: list of (name, [values per group]); None = missing."""
    L, R, T, B = 48, 16, 26, 44
    pw, ph = w - L - R, h - T - B
    s = svg_open(w, h)
    for i in range(0, 6):
        y = T + ph - ph * i / 5; v = ymax * i / 5
        s.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' stroke='#e2e2e2'/>")
        s.append(text(L - 6, y + 4, fmt.format(v), 9.5, "end", fill="#555"))
    if band:
        lo, hi = band; y1 = T + ph - ph * hi / ymax; y2 = T + ph - ph * lo / ymax
        s.append(f"<rect x='{L}' y='{y1:.1f}' width='{pw}' height='{y2-y1:.1f}' fill='#c9a227' fill-opacity='0.12'/>")
    n = len(groups); k = len(series); gw = pw / n; bw = gw * 0.7 / k
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + gw * 0.15
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            if v is None:
                s.append(text(x0 + si * bw + bw / 2, T + ph - 6, "—", 10, "middle", fill="#999")); continue
            bh = ph * v / ymax
            s.append(f"<rect x='{x0 + si*bw:.1f}' y='{T + ph - bh:.1f}' width='{bw - 2:.1f}' height='{bh:.1f}' fill='{colors[si % len(colors)]}'/>")
            s.append(text(x0 + si * bw + bw / 2 - 1, T + ph - bh - 4, fmt.format(v), 9, "middle", fill="#333"))
        s.append(text(L + gi * gw + gw / 2, T + ph + 16, g, 10.5, "middle"))
    s.append(text(14, T + ph / 2, ylabel, 10, "middle", fill="#444", rot=-90))
    if legend:
        lx = L + 4
        for si, (name, _) in enumerate(series):
            s.append(f"<rect x='{lx}' y='{h-12}' width='10' height='10' fill='{colors[si % len(colors)]}'/>")
            s.append(text(lx + 14, h - 3, name, 10)); lx += 14 + 7 * len(name) + 16
    s.append("</svg>"); path.write_text("\n".join(s))

def stacked_bars(path, groups, parts, ymax, ylabel, w=520, h=250, colors=("#2f7fb3", "#9dc3e0")):
    L, R, T, B = 52, 16, 24, 44; pw, ph = w - L - R, h - T - B
    s = svg_open(w, h)
    for i in range(0, 6):
        y = T + ph - ph * i / 5
        s.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' stroke='#e2e2e2'/>")
        s.append(text(L - 6, y + 4, f"{ymax*i/5/1000:.0f}k", 9.5, "end", fill="#555"))
    n = len(groups); gw = pw / n; bw = gw * 0.5
    for gi, g in enumerate(groups):
        x = L + gi * gw + (gw - bw) / 2; y = T + ph
        for pi, (name, vals) in enumerate(parts):
            v = vals[gi]; bh = ph * v / ymax; y -= bh
            s.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{bh:.1f}' fill='{colors[pi]}'/>")
            if bh > 14: s.append(text(x + bw / 2, y + bh / 2 + 4, f"{v/1000:.1f}k", 9.5, "middle", fill="#fff" if pi == 0 else "#123"))
        s.append(text(x + bw / 2, y - 5, f"{sum(p[1][gi] for p in parts)/1000:.1f}k", 10, "middle", weight="bold"))
        s.append(text(L + gi * gw + gw / 2, T + ph + 16, g, 10.5, "middle"))
    s.append(text(14, T + ph / 2, ylabel, 10, "middle", fill="#444", rot=-90))
    lx = L + 4
    for pi, (name, _) in enumerate(parts):
        s.append(f"<rect x='{lx}' y='{h-12}' width='10' height='10' fill='{colors[pi]}'/>")
        s.append(text(lx + 14, h - 3, name, 10)); lx += 14 + 7 * len(name) + 16
    s.append("</svg>"); path.write_text("\n".join(s))

def lines(path, xs, series, ylabel, w=520, h=256, ymax=100, ymin=50, colors=PAL, events=()):
    L, R, T, B = 48, 16, 22, 56; pw, ph = w - L - R, h - T - B
    s = svg_open(w, h)
    steps = 5
    for i in range(steps + 1):
        v = ymin + (ymax - ymin) * i / steps; y = T + ph - ph * i / steps
        s.append(f"<line x1='{L}' y1='{y:.1f}' x2='{L+pw}' y2='{y:.1f}' stroke='#e2e2e2'/>")
        s.append(text(L - 6, y + 4, f"{v:.0f}", 9.5, "end", fill="#555"))
    def X(i): return L + pw * (i + 0.5) / len(xs)
    def Y(v): return T + ph - ph * (v - ymin) / (ymax - ymin)
    for e in events:
        i = xs.index(e); s.append(f"<line x1='{X(i):.1f}' y1='{T}' x2='{X(i):.1f}' y2='{T+ph}' stroke='#c9a227' stroke-dasharray='3 3'/>")
        s.append(text(X(i), T - 6, "event", 8.5, "middle", fill="#8a6d0f"))
    for si, (name, vals) in enumerate(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        s.append(f"<polyline points='{pts}' fill='none' stroke='{colors[si]}' stroke-width='2'/>")
        for i, v in enumerate(vals):
            if v is not None: s.append(f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='3' fill='{colors[si]}'/>")
    for i, x in enumerate(xs): s.append(text(X(i), T + ph + 16, str(x), 10, "middle"))
    s.append(text(L + pw / 2, h - 18, "session", 10, "middle", fill="#444"))
    s.append(text(14, T + ph / 2, ylabel, 10, "middle", fill="#444", rot=-90))
    lx = L + 4
    for si, (name, _) in enumerate(series):
        s.append(f"<line x1='{lx}' y1='{h-6}' x2='{lx+14}' y2='{h-6}' stroke='{colors[si]}' stroke-width='2'/>")
        s.append(text(lx + 18, h - 2, name, 10)); lx += 18 + 6.5 * len(name) + 18
    s.append("</svg>"); path.write_text("\n".join(s))

# ---------- data ----------
VERS = [("v1", "v1"), ("v2", "v2"), ("v3", "qwen36-4bit")]
ARMS = ["summary", "auto", "minimal", "full"]
book_pct = {v: {a: mean_pct(book_runs(lab, a)) for a in ARMS} for v, lab in VERS}
pong_pct = {}
for v, lab in VERS:
    pong_pct[v] = {}
    for a in ["control", "auto", "minimal", "full"]:
        rows = pong_runs(lab, a); s = sum(x[1] for x in rows)
        pong_pct[v][a] = 100 * sum(x[0] for x in rows) / s if s else None
print("book", json.dumps({v: {a: round(x, 1) for a, x in d.items() if x is not None} for v, d in book_pct.items()}))
print("pong", json.dumps({v: {a: round(x, 1) for a, x in d.items() if x is not None} for v, d in pong_pct.items()}))

# Fig 3: book carry-over by arm and version, noise band 84-96 (spread of the unchanged summary arm)
grouped_bars(OUT / "fig_book_arms.svg", ["summary (memory off)", "auto (engine writes)", "minimal", "full"],
             [(v, [book_pct[v][a] for a in ARMS]) for v, _ in VERS], 100, "carry-over, sessions 2–10 (%)",
             fmt="{:.0f}", band=(84, 96))
# Fig 4: pong
grouped_bars(OUT / "fig_pong_arms.svg", ["control (memory off)", "auto", "minimal", "full"],
             [(v, [pong_pct[v][a] for a in ["control", "auto", "minimal", "full"]]) for v, _ in VERS], 100,
             "rules reproduced in Python and C (%)", fmt="{:.0f}")
# Fig 5: cost split for auto
cost = {}
for v, lab in VERS:
    rows = book_runs(lab, "auto"); n = len(rows)
    cost[v] = dict(req=sum(r["req_p"] for r in rows) / n, con=sum(r["con_p"] for r in rows) / n,
                   comp=sum(r["req_c"] + r["con_c"] for r in rows) / n, sec=sum(r["sec"] for r in rows) / n)
srows = {v: book_runs(lab, "summary") for v, lab in VERS}
for v in cost: cost[v]["summary_req"] = sum(r["req_p"] for r in srows[v]) / len(srows[v]); cost[v]["summary_sec"] = sum(r["sec"] for r in srows[v]) / len(srows[v])
print("cost", json.dumps({v: {k: round(x) for k, x in d.items()} for v, d in cost.items()}))
stacked_bars(OUT / "fig_cost.svg", ["v1 auto", "v2 auto", "v3 auto"],
             [("request prompts", [cost[v]["req"] for v, _ in VERS]), ("consolidation prompts", [cost[v]["con"] for v, _ in VERS])],
             35000, "prompt tokens per book run")
# Fig 6: per-session, v3 summary vs auto (mean of 3 runs), sessions 2-10
def per_session(label, arm):
    rows = book_runs(label, arm); out = []
    for sess in range(2, 11):
        c = sum(r["per"].get(sess, (0, 0))[0] for r in rows); t = sum(r["per"].get(sess, (0, 0))[1] for r in rows)
        out.append(100 * c / t if t else None)
    return out
lines(OUT / "fig_sessions.svg", list(range(2, 11)),
      [("summary (memory off)", per_session("qwen36-4bit", "summary")), ("auto, Qwen 3.6 4-bit", per_session("qwen36-4bit", "auto")),
       ("auto, Qwen 3.6 8-bit", per_session("qwen36-8bit", "auto"))],
      "quiz correct (%)", events=(2, 4, 6, 8, 9), colors=["#5b6b7f", "#2f7fb3", "#d98f2b"])
# Fig 7: cross-install
inst = [("Qwen 3.6 4-bit", "qwen36-4bit"), ("Qwen 3.6 8-bit", "qwen36-8bit"), ("Ornith 1.5 4-bit", "ornith-4bit"),
        ("Ornith 1.5 8-bit", "ornith-8bit"), ("AgentWorld 4-bit", "agentworld-4bit"), ("AgentWorld 8-bit", "agentworld-8bit")]
off = [mean_pct(book_runs(lab, "summary")) if book_runs(lab, "summary") else None for _, lab in inst]
on = [mean_pct(book_runs(lab, "auto")) if book_runs(lab, "auto") else None for _, lab in inst]
print("installs", list(zip([n for n, _ in inst], off, on)))
grouped_bars(OUT / "fig_installs.svg", [n.replace(" ", " ") for n, _ in inst], [("memory off", off), ("memory on (auto)", on)], 100,
             "book carry-over (%)", w=560, band=None, colors=["#5b6b7f", "#2f7fb3"])
# Fig 2: the write is the unreliable link (evaluation 2026-09-06 single runs + v1 auto)
grouped_bars(OUT / "fig_writer.svg", ["model writes (minimal)", "model writes (full)", "forced summary", "engine consolidation (v1 auto)"],
             [("carry-over", [100 * 39 / 126, 100 * 54 / 126, 100 * 108 / 126, book_pct["v1"]["auto"]])], 100,
             "book carry-over (%)", w=520, h=230, legend=False, colors=["#2f7fb3"])
print("figures written to", OUT)
