# Release and build rules — TinyTitan

The release and build standard for this repository.

**Part 1 is generic** and identical in every Pummelchen repository. **Part 2 is this
repository's own section**, and it wins wherever the two disagree.

This file is a **generated copy** — do not edit it here. *For maintainers:* the
master is `docs/release-rules.md` in the `TinyTitan` repository, which holds Part 1
once and every repository's Part 2 side by side; edit that and run
`tools/sync-release-rules.py --apply`. An agent working in this repository should
treat this file as authoritative and does not need to leave the repository.

---

# Part 1 — Generic rules

## 1.1 Scope

These apply to any repository that produces a **runnable artifact**: a binary, a
library, an image, a package. Repositories that only hold documents, data or
configuration are out of scope, and should say so in their Part 2 section rather
than adopting a release process they cannot use.

## 1.2 Non-negotiable

1. **Apple Silicon only.** Build native `arm64`. This covers M1–M6. Never
   `--arch x86_64`, never `ARCHS=arm64 x86_64`, and never `lipo -create` — that
   is how a universal binary gets made, and there is no x86_64 build.
2. **Assert it, do not assume it.** After building, check the artifact:
   `lipo -archs <binary>` must be exactly `arm64`. A build that silently produced
   a fat binary is a release defect, not a build option.
3. **Every release carries the artifacts.** A tag alone is not a release. If the
   Release page has no binaries attached, the release did not happen.
4. **No hardcoded build-toolchain triple in a path.** `.build/release` is the
   stable spelling. `.build/arm64-apple-macosx/release` points at nothing on a
   newer toolchain and at a stale binary on this one. The one exception is a build
   that explicitly passes `--arch arm64`: then the triple directory really is
   where SwiftPM writes, and that build must also assert the arch (§1.2.2).
5. **One checksummed artifact per target, or one checksum file covering all of
   them.** Never publish a binary without a digest beside it.
6. **Dry run by default; publish only on an explicit flag.**
7. **Never fetch a model, dataset or dependency to make a gate pass.** A check
   that cannot run is reported *not checked* — and the release notes must name it.
   "Not checked, no input" and "checked and identical" are different sentences.

## 1.3 Identity

The version or build number is **single-sourced and enforced**, not maintained by
hope.

- **One authoritative value.** A file at the repository root — `VERSION` for a
  semantic version, `BUILD_NUMBER` for a build number. Anywhere else it appears
  is a **mirror**, and the build or CI must fail when a mirror disagrees.
- **Pick one scheme and state it.** Semantic versions (`vX.Y.Z`) or build numbers
  (`b1`, `b2`). Do not mix them, and do not "helpfully" introduce versions into a
  project that uses build numbers.
- **The build refuses a malformed or inconsistent identity.** Fail at configure
  or compile time, not at release time.
- **Identity is observable.** A user must be able to say what they are running
  from the artifact alone: the archive filename, or the program's own answer, or
  both.
- **Bump once, propagate mechanically.** Provide a command that writes the mirrors
  from the authoritative value. A release is one edit plus one command.
- **A second declaration in a test is a defect.** Derive the expected value from
  the source of truth; a literal in a test means every bump fails a test that is
  not about the version, and the tempting fix — editing the test — is how a wrong
  version ships.
- **Multi-library projects version in lockstep.** Libraries that ship together and
  interoperate carry the **same** version, because a caller pairing them has no
  other way to know the pair is compatible. A library with no code change is
  recompiled and republished at the new number rather than left behind.
  Lockstep applies to the **library version only** — an ABI version, protocol
  draft, or schema version is a separate axis and must not be dragged along.

## 1.4 Preconditions

Before starting, confirm and record: the OS floor and toolchain floor are met
(`sw_vers`, `swift --version`); there is disk for a clean scratch build plus the
staged archive; `memory_pressure -Q` is acceptable; **no competing build or model
process is running**; `gh auth status` is the repository owner's account; the tree
is clean; and `HEAD` **is** the tag.

**Never terminate a process you did not start.** If one is blocking, name it with
its parent and age, and stop.

## 1.5 Gates

Run these in order, and make each one **able to fail**:

1. **Lint** — the project's own lint gates.
2. **Full test suite**, serially, and it must report the count that passed.
3. **Parity or golden checks** — real inference, real rendering, real protocol
   frames; whatever "the output is unchanged" means for this project.
4. **A clean scratch build** with the log scanned for warnings.

Two traps, both of which have shipped broken gates in this organisation:

- **A gate that cannot fail is not a gate.** A guard that looks for a file the
  build never produces passes for every input. A warning scan over an *incremental*
  build compiles nothing and passes vacuously — always use a fresh scratch path.
  Before trusting a new gate, break its input and watch it fail.
- **Guard the plan, not the byproduct.** Ask the build system what it resolved
  (`swift package describe --type json`, `cmake --build ... -t help`) rather than
  checking for artifacts after the fact.

## 1.6 Packaging

The archive contains, at minimum:

- the **executables or libraries**, built for arm64;
- **resource bundles** — a Swift binary without its `.bundle` cannot load its
  Metal kernels, and this fails at runtime rather than at build time;
- `LICENSE`, and `NOTICE` / `THIRD_PARTY_NOTICES.md` where third-party code is
  redistributed;
- a **`README-binaries.txt`** stating the platform floor, that the build is
  Apple-Silicon-only, and that the binaries are **not code-signed or notarized** —
  with the quarantine command (`xattr -dr com.apple.quarantine <path>`) so a user
  who verified the checksum can run them. Do not imply a notarized build.

Name the archive `<project>[-<library>]-<version>-macos-arm64.tar.gz`; the
library segment is required only for a multi-library project, and exists so two
artifacts of the same release are distinguishable.

## 1.7 Publishing

```bash
gh release create "$TAG" "$ARCHIVE" "$ARCHIVE.sha256" \
  --repo <owner>/<repo> --title "<Project> $VERSION" \
  --notes-file "$NOTES" --latest
```

**Pin `--repo` on every `gh` call.** In a fork `gh` defaults to the *parent*
repository, so `gh release list` shows another project's releases and
`gh release create` fails with a misleading "tag has not been pushed".

## 1.8 Release notes

- Full notes in `docs/release-notes-vX.Y.md` (or the repository's equivalent),
  one section per user-visible change, each naming the check that backs it.
- End with a checksum block carrying `SHA256_PENDING` and
  `ARCHIVE_BYTES_PENDING`, substituted at publish time. **Never copy a size out
  of a dry run** — publish rebuilds, and the archive differs.
- `--publish` must **refuse** unless the notes carry the placeholder or quote the
  real value. A release quoting the wrong digest is worse than one quoting none.
- Name **every** check that did not run, and why.
- The README gets **no release callout**. It changes only when a fact it states
  changes. The changelog is the announcement.

## 1.9 After publishing

Verify the Release: the notes quote the digest in the `.sha256` beside it, the
assets are the archive and its checksum, and the changelog points at the same tag.
Leave previous releases' notes and performance tables alone.

## 1.10 Cross-repository

- **This file is the master; every repository's copy is generated from it.** The
  master is `docs/release-rules.md` in `TinyTitan`, and its
  `tools/sync-release-rules.py` splits it into Part 1 and each repository's Part 2
  and deploys the resulting `RELEASE.md` plus the `## Releasing` section of each
  `AGENTS.md`. Edit the master and run `--apply`; `--check` is the drift gate and
  exits non-zero when a committed copy no longer matches. **Never hand-edit a
  deployed copy** — the next run overwrites it.
- **Repository rules live in the master, not in shell-script comments.** A rule an
  agent cannot find is a rule that will be broken.
- **`AGENTS.md` is the one instruction file, and every harness must reach it.** This
  account works with Codex, Claude Code, DeepSeek Harness, OpenCode, Qwen Code,
  Qoder and Zed. Six read `AGENTS.md` directly; **Claude Code does not** — its
  documentation is explicit that it reads `CLAUDE.md`, not `AGENTS.md` — so every
  repository also carries a committed `CLAUDE.md` whose entire content is the
  `@AGENTS.md` import. Commit it: a symlink made on one machine is invisible to a
  fresh clone, to CI and to every other checkout, and on Windows it needs
  Administrator rights. Qwen Code reads `AGENTS.md` alongside its own `QWEN.md`, so
  there is nothing to duplicate for it.
- **Never add a file that shadows `AGENTS.md`.** Zed takes the *first match* from
  `.rules`, `.cursorrules`, `.windsurfrules`, `.clinerules`,
  `.github/copilot-instructions.md`, `AGENT.md`, and only then `AGENTS.md` — so any
  of those six silently replaces this file for every Zed user.
  `tools/sync-release-rules.py --check` fails when one appears.
- **An archived repository is read-only.** Nothing can be committed to it, so no
  release step may depend on one. Name the exclusion rather than leaving a gap.
- **A check that has never been seen to fail is not yet trusted.**

---

# Part 2 — This repository

## TinyTitan — Swift, semantic version, 21 releases

*The reference implementation of this standard.* Runbook:
`docs/release-process.md`. Mechanism: `tools/release.sh`.

### Fork status — not upstreamed, deliberately not detached

This repository is a GitHub **fork** of `drumih/turbo-fieldfare`, and it is
**deliberately left as a fork** — do not detach it from the fork network. Leaving
is permanent, and the standalone repository would not retain its wiki, issues, pull
requests, stars, watchers or child forks; the wiki alone holds 15 pages across 223
commits, alongside 21 releases.

The fork relationship is inert. Nothing here is upstreamed: **no pull requests,
cherry-picks or patches go to the parent**, and every change stays inside this
repository.

- **Identity** `vX.Y.Z`. The only version literal in the tree is
  `CFBundleVersion` / `CFBundleShortVersionString` in `tools/install_tinytitan.sh`;
  the wiki `Changelog.md` carries the announcement and the README carries none.
- **Artifacts** `tinytitan-X.Y-macos-arm64.tar.gz` + `.sha256`, containing **six
  executables** — `TinyTitanServer`, `TinyTitanCLI`, `TinyTitanMac`,
  `TinyTitanDecodeService`, `TinyTitanRepack`, `TinyTitanBench` — plus the
  `.bundle` resources, licence and notices.
- **Gates** `tools/lint.sh` (five gates: force-cast, func-length, sendable,
  converter, arch-path); `swift test --no-parallel`; **every installed model with
  a golden target**, through `tools/golden-baseline.sh --check`; then a clean
  scratch build with the warning scan.
- **Mandatory** `tools/internal-speeds.py --record --label vX.Y --baseline …`.
  Any metric or duration past a 10% regression blocks the release until fixed or
  explained in `### Verification`. The record is committed with the release.
- **Only models already installed under `models/` are verified**, and `release.sh`
  fingerprints the install set so a gate cannot install one to go green. Missing
  installs are reported *not checked* and must be named in the notes.
- **Traps** a golden gate that *refused to start* is reported as a "mismatch" —
  read the line above it. A synced-folder install can be online-only, which
  surfaces as `parallel expert read failed`.
- The model install and the checkout are separate jobs; adding a model is
  `docs/adding-a-model.md` and an operator decision, never a release side-effect.
