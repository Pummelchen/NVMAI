#!/usr/bin/env python3
"""Regenerate every repository's RELEASE.md from docs/release-rules.md.

Agents are not allowed to read across repositories, so each repository carries
its own copy of the release rules. That copy is *generated* from the master in
this repository, so a rule changes in exactly one place:

    docs/release-rules.md
      # Part 1 — Generic rules      <- applies to every repository
      # Part 2 — Per repository
        ## <Repo> — <tagline>       <- applies to that repository alone

Usage
-----
    python3 tools/sync-release-rules.py --check      # drift gate: exit 1 on drift
    python3 tools/sync-release-rules.py --dry-run    # show what would change
    python3 tools/sync-release-rules.py --emit DIR   # write the files into DIR
    python3 tools/sync-release-rules.py --apply      # push branches + open PRs

`--apply` never writes to a default branch: it force-resets a single shared
branch per repository from that repository's default branch, commits the
generated files, and opens or updates the pull request.

Two of these repositories are forks (TinyTitan, OpenRA) and in a fork `gh`
defaults to the *parent* repository, so every call here names the repository
explicitly. Do not "simplify" that away.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys

OWNER = "Pummelchen"
BRANCH = "docs/release-rules"
MASTER_REL = "docs/release-rules.md"

# The repository that holds the master. Its own branch must also carry the
# master file, since that branch is where the master is edited.
MASTER_REPO = "TinyTitan"

ARCHIVED = {"FXAI", "FXAI-V0"}

PART2_HEAD = "# Part 2 — Per repository"
SEP = "\n\n---\n\n"

BEGIN = "<!-- release-rules:begin -->"
END = "<!-- release-rules:end -->"

PREAMBLE = """The release and build standard for this repository.

**Part 1 is generic** and identical in every {owner} repository. **Part 2 is this
repository's own section**, and it wins wherever the two disagree.

This file is a **generated copy** — do not edit it here. *For maintainers:* the
master is `{master_rel}` in the `{master_repo}` repository, which holds Part 1
once and every repository's Part 2 side by side; edit that and run
`tools/sync-release-rules.py --apply`. An agent working in this repository should
treat this file as authoritative and does not need to leave the repository."""


# --------------------------------------------------------------------------
# master -> pieces
# --------------------------------------------------------------------------

def split_master(master: str):
    """Return (part1, intro, {repo: section}) from the master document."""
    if PART2_HEAD not in master:
        sys.exit(f"master is missing the {PART2_HEAD!r} heading")
    i = master.index(PART2_HEAD)
    part1 = master[:i].rstrip()
    if part1.endswith("---"):
        part1 = part1[:-3].rstrip()
    if not part1:
        sys.exit("master has an empty Part 1")

    rest = master[i + len(PART2_HEAD):]
    m = re.search(r"(?m)^## ", rest)
    if not m:
        sys.exit("master has no '## <Repo> — ...' sections under Part 2")
    intro = rest[: m.start()].strip()

    sections: dict[str, str] = {}
    for chunk in re.split(r"(?m)^(?=## )", rest[m.start():]):
        chunk = chunk.rstrip()
        if not chunk:
            continue
        title = chunk.splitlines()[0][3:].strip()
        key_part = title.split(" — ")[0]
        for name in (n.strip() for n in key_part.split(",")):
            if name:
                sections[name] = chunk
    return part1, intro, sections


def build_release_md(repo: str, part1: str, section: str) -> str:
    preamble = PREAMBLE.format(owner=OWNER, master_rel=MASTER_REL, master_repo=MASTER_REPO)
    return (
        f"# Release and build rules — {repo}\n\n"
        f"{preamble}\n\n"
        f"---\n\n"
        f"{part1}\n\n"
        f"---\n\n"
        f"# Part 2 — This repository\n\n"
        f"{section}\n"
    )


def agents_section() -> str:
    return f"""{BEGIN}
## Releasing

**Read [`RELEASE.md`](RELEASE.md) before cutting a release.** It carries the
generic rules every {OWNER} repository follows, plus this repository's own
section. Do not improvise a release.

The non-negotiables:

- **Apple Silicon only** — build native `arm64` (M1–M6). Never `--arch x86_64`,
  never `ARCHS=arm64 x86_64`, and never `lipo -create`, which is how a universal
  binary gets made.
- **Assert it** — `lipo -archs <binary>` must report exactly `arm64`. A build that
  silently produced a fat binary is a release defect, not a build option.
- **Every release carries the artifacts.** A tag alone is not a release.
- **Identity is single-sourced and enforced** — never bump one declaration of the
  version or build number on its own; the build or CI must fail on a mismatch.
- **Dry run first**; publish only on an explicit flag.
- **Never fetch a model, dataset or dependency to make a gate pass.** A check that
  cannot run is reported *not checked*, and the release notes must name it.
{END}"""


def build_agents_md(repo: str, existing: str | None) -> str:
    """Insert or replace the marked release section, leaving the rest alone."""
    block = agents_section()
    if existing is None:
        return f"# {repo}\n\n{block}\n"
    if BEGIN in existing and END in existing:
        pre = existing[: existing.index(BEGIN)].rstrip()
        post = existing[existing.index(END) + len(END):].strip()
        out = f"{pre}\n\n{block}"
        return out + (f"\n\n{post}\n" if post else "\n")
    return existing.rstrip() + "\n\n" + block + "\n"


# --------------------------------------------------------------------------
# github
# --------------------------------------------------------------------------

def gh(args, **kw):
    return subprocess.run(["gh"] + args, capture_output=True, text=True, **kw)


def get_file(repo: str, path: str, ref: str):
    r = gh(["api", f"repos/{OWNER}/{repo}/contents/{path}?ref={ref}"])
    if r.returncode != 0:
        return None
    d = json.loads(r.stdout)
    return base64.b64decode(d["content"]).decode(), d["sha"]


def put_file(repo: str, path: str, text: str, message: str, branch: str, sha=None):
    payload = {
        "message": message,
        "content": base64.b64encode(text.encode()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    r = gh(["api", "-X", "PUT", f"repos/{OWNER}/{repo}/contents/{path}", "--input", "-"],
           input=json.dumps(payload))
    return r.returncode == 0, r.stderr.strip()[:200]


def reset_branch(repo: str, base: str, head_sha: str) -> bool:
    """Force the shared branch back to the default branch, so a re-run is clean."""
    r = gh(["api", "-X", "PATCH", f"repos/{OWNER}/{repo}/git/refs/heads/{BRANCH}",
            "-f", f"sha={head_sha}", "-F", "force=true"])
    if r.returncode == 0:
        return True
    if "Not Found" in (r.stderr + r.stdout):
        c = gh(["api", "-X", "POST", f"repos/{OWNER}/{repo}/git/refs",
                "-f", f"ref=refs/heads/{BRANCH}", "-f", f"sha={head_sha}"])
        return c.returncode == 0
    print(f"    reset failed: {r.stderr.strip()[:160]}", file=sys.stderr)
    return False


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="exit 1 if any repository has drifted")
    g.add_argument("--dry-run", action="store_true", help="report what would change")
    g.add_argument("--emit", metavar="DIR", help="write generated files into DIR")
    g.add_argument("--apply", action="store_true", help="push branches and open PRs")
    ap.add_argument("--master", default=None, help=f"path to the master (default {MASTER_REL})")
    ap.add_argument("--reset", action="store_true",
                    help="force each shared branch back to its default branch first "
                         "(bootstrap only — this churns the pull requests)")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    master_path = pathlib.Path(args.master) if args.master else root / MASTER_REL
    if not master_path.exists():
        sys.exit(f"no master at {master_path} — run from a checkout of {MASTER_REPO}")
    master = master_path.read_text()
    part1, intro, sections = split_master(master)

    # The repositories to deploy to are the master's own Part 2 sections, minus
    # the archived ones. This makes adding a repository a master-only edit.
    repos = [n for n in sections if n not in ARCHIVED]
    if MASTER_REPO not in repos:
        sys.exit(f"the master has no '## {MASTER_REPO} — ...' section")

    if args.emit:
        out = pathlib.Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        for repo in repos:
            (out / f"{repo}.RELEASE.md").write_text(build_release_md(repo, part1, sections[repo]))
            (out / f"{repo}.AGENTS.md").write_text(build_agents_md(repo, None))
        print(f"emitted {len(repos)} repositories into {out}")
        return 0

    print(f"master: {master_path}  ({len(master.splitlines())} lines)")
    print(f"repositories: {len(repos)}  ({', '.join(repos)})")
    print(f"archived (skipped): {', '.join(sorted(ARCHIVED))}\n")

    drift, failed = [], []
    for repo in repos:
        meta = gh(["api", f"repos/{OWNER}/{repo}"])
        if meta.returncode != 0:
            print(f"  {repo:<22} NO ACCESS")
            failed.append(repo)
            continue
        base = json.loads(meta.stdout)["default_branch"]

        want_release = build_release_md(repo, part1, sections[repo])

        if args.check:
            # Prefer the open PR branch while one exists, so the gate is
            # meaningful before the change has landed; otherwise check what is
            # actually committed on the default branch.
            ref = base
            pr = gh(["pr", "list", "-R", f"{OWNER}/{repo}", "--head", BRANCH,
                     "--state", "open", "--json", "url", "--jq", '.[0].url // ""'])
            if pr.returncode == 0 and pr.stdout.strip():
                ref = BRANCH
            rel = get_file(repo, "RELEASE.md", ref)
            ag = get_file(repo, "AGENTS.md", ref)
            rel_ok = rel is not None and rel[0] == want_release
            ag_ok = ag is not None and BEGIN in ag[0] and "lipo -create" in ag[0]
            if not (rel_ok and ag_ok):
                drift.append(repo)
            print(f"  {repo:<22} {'ok' if rel_ok and ag_ok else 'DRIFT':<6} {ref:<18}"
                  f" RELEASE.md{'' if rel_ok else ' stale/missing'}"
                  f"   AGENTS.md{'' if ag_ok else ' stale/missing'}")
            continue

        # Prefer the branch's own AGENTS.md so the repository's own body is
        # preserved; only the marked release block is ever rewritten. Reading the
        # default branch instead would overwrite a body that exists only on the
        # branch with the bare stub.
        agents = get_file(repo, "AGENTS.md", BRANCH)
        if agents is None:
            agents = get_file(repo, "AGENTS.md", base)
        want_agents = build_agents_md(repo, agents[0] if agents else None)

        if args.dry_run:
            have = get_file(repo, "RELEASE.md", base)
            release_ok = have is not None and have[0] == want_release
            print(f"  {repo:<22} RELEASE.md {'unchanged' if release_ok else 'WRITE'}"
                  f"   AGENTS.md {'marked' if BEGIN in (agents[0] if agents else '') else 'WRITE'}")
            continue

        # --apply
        head = gh(["api", f"repos/{OWNER}/{repo}/commits/{base}"])
        if head.returncode != 0:
            print(f"  {repo:<22} cannot read {base}")
            failed.append(repo)
            continue
        head_sha = json.loads(head.stdout)["sha"]

        if args.reset:
            if not reset_branch(repo, base, head_sha):
                failed.append(repo)
                continue
        else:
            # Create the branch if it is missing; otherwise update in place, so a
            # routine run does not churn the pull request.
            br = gh(["api", "-X", "POST", f"repos/{OWNER}/{repo}/git/refs",
                     "-f", f"ref=refs/heads/{BRANCH}", "-f", f"sha={head_sha}"])
            if br.returncode != 0 and "already exists" not in (br.stderr + br.stdout):
                print(f"  {repo:<22} branch FAILED {br.stderr.strip()[:120]}")
                failed.append(repo)
                continue

        _, rel_sha = get_file(repo, "RELEASE.md", BRANCH) or (None, None)
        ok1, err1 = put_file(repo, "RELEASE.md", want_release,
                             "docs: regenerate the release and build rules", BRANCH, rel_sha)
        _, ag_sha = get_file(repo, "AGENTS.md", BRANCH) or (None, None)
        ok2, err2 = put_file(repo, "AGENTS.md", want_agents,
                             "docs: point AGENTS.md at the release rules", BRANCH, ag_sha)

        if repo == MASTER_REPO:
            _, mk_sha = get_file(repo, MASTER_REL, BRANCH) or (None, None)
            ok3, err3 = put_file(repo, MASTER_REL, master,
                                 "docs: the master release and build rules", BRANCH, mk_sha)
        else:
            ok3, err3 = True, ""

        existing = gh(["pr", "list", "-R", f"{OWNER}/{repo}", "--head", BRANCH,
                       "--state", "open", "--json", "url", "--jq", '.[0].url // ""'])
        if existing.returncode == 0 and existing.stdout.strip():
            url = existing.stdout.strip()
        else:
            body = (f"Regenerates this repository's copy of the release and build "
                    f"rules from the master in `{MASTER_REPO}/{MASTER_REL}`.\n\n"
                    f"- **`RELEASE.md`** — Part 1 (generic, identical everywhere) plus "
                    f"Part 2, this repository's own section.\n"
                    f"- **`AGENTS.md`** — the discovery point, carrying the "
                    f"non-negotiables and a pointer to `RELEASE.md`.\n\n"
                    f"Generated, not hand-written: edit the master and run "
                    f"`tools/sync-release-rules.py --apply`.")
            bf = pathlib.Path(f"/tmp/_sync_pr_{repo}.md")
            bf.write_text(body)
            r = gh(["pr", "create", "-R", f"{OWNER}/{repo}", "--base", base,
                    "--head", BRANCH, "--title",
                    "docs: the release and build rules for this repository",
                    "--body-file", str(bf)])
            url = (r.stdout.strip().splitlines()[-1] if r.returncode == 0
                   else f"PR FAILED: {r.stderr.strip()[:120]}")

        good = ok1 and ok2 and ok3
        if not good:
            failed.append(repo)
        print(f"  {repo:<22} {'ok' if good else 'FAIL'}   {url}"
              + ("" if good else f"   {err1} {err2} {err3}"))

    if args.check:
        print()
        if drift or failed:
            print(f"DRIFT: {len(drift)} stale, {len(failed)} unreadable — "
                  f"run tools/sync-release-rules.py --apply")
            return 1
        print(f"all {len(repos)} repositories match the master")
        return 0
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
