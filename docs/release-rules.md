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

# Part 2 — Per repository

Active means not archived. `FXAI` and `FXAI-V0` are archived and out of scope
until unarchived, and an archived repository cannot receive commits, so no gate
may depend on them.

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

## WebTransport — Swift **and** C99, semantic version, 11 releases

**One repository, two libraries, one version.** This is the project the lockstep
rule exists for.

- **Identity** semantic version, single-sourced from `VERSION` at the root.
  Mirrors: `WT_VERSION_*` in `C99/include/webtransport/version.h` and
  `library` in `Swift/Sources/WebTransport/WebTransportVersion.swift`.
  Enforced by `Swift/check-version-sync.sh` in CI and by the C99 CMake configure.
  Bump with `./Swift/check-version-sync.sh --write`. **The Swift and C99 libraries
  always carry the same number**; if only one changed, recompile the other at the
  new number rather than leaving it behind.
- **`WT_ABI_VERSION` is not part of the lockstep.** It moves only for a breaking
  layout or signature change; a bug-fix release moves the version and not the ABI.
  `wt_protocol_draft()` is a third, separate axis.
- **C99 artifacts** built by `C99/platform/macos26/compile-dylib.sh` (CMake,
  Ninja); `C99/platform/{debian,freebsd}/compile-so.sh` for the other platforms.
- **Swift artifacts** built by `Swift/build-release-apple-silicon.sh`, which is
  the most rigorous build in the organisation and the model for the rest:
  - `swift package describe --type json` is checked for experiment/spike targets
    **before** building, because a file-existence guard "could not fail for any
    input";
  - **two full build passes** with `SOURCE_DATE_EPOCH=0`,
    `SWIFT_DETERMINISTIC_HASHING=1`, `ZERO_AR_DATE=1`, compared by a **normalized
    Mach-O hash**, so the release is reproducible;
  - `lipo -archs` must report exactly `arm64`;
  - output lands in `.build/release-artifacts/` with a `SHA256SUMS`.
- **Gates** `Swift/check-toolchain.sh 6.4 27.0`, `Swift/check-manifest-sync.sh`
  (19 shared targets must agree across the two manifests),
  `Swift/check-version-sync.sh`, `check-api-compatibility.sh`, the C99
  `C99/scripts/check-*.sh` family, and the full suite under ASan and TSan.
- **Two manifests** — the root `Package.swift` and `Swift/Package.swift` —
  intentionally expose different product sets; shared targets must not diverge.
- **Publishing** currently ships only the Swift products. The C99 library is new
  and not yet built for release; when it joins, it joins **this** tag and these
  notes rather than getting its own, and the notes must say which library is not
  yet built.

## XAIOS — C, **build numbers**, 6 releases

*An operating system, not an application. Two rules that apply elsewhere do not
apply here.*

- **Identity is a build number, not a version.** Tags are `b1`, `b2`, … `bN`, and
  the Release title is "XAIOS Build N". **Do not introduce semantic versions.**
  Single-sourced from the `BUILD_NUMBER` file at the root, which must be a whole
  number — the build refuses to proceed otherwise — and which the running system
  reports on its first boot line and from `xaiosctl version`, so a support case,
  an advisory and a file on disk cannot disagree about what is running.
- **The multi-architecture release is intentional and stays as it is.** XAIOS
  publishes guest images for `aarch64`, `riscv64` **and** `x86_64`. Those are
  **guest** architectures, not host binaries: the Apple-Silicon-only rule (§1.2.1)
  governs host binaries a contributor builds for their own Mac, and does not
  restrict which architectures the OS ships images for. Do not "standardise" this
  list, and do not apply a host `lipo` assertion to a guest image.
- **Artifacts** are committed under `release/` (`xaios_bN-<arch>.iso.zip`) beside a
  per-build notes file `release/xaios_bN.md`, and the same files are attached to
  the Release. This is the one project that keeps built artifacts in-tree; that is
  a deliberate choice, not an oversight, and the reason is that a released image
  must remain fetchable at the tag it names.
- **Builders** `scripts/build-arch-image.sh` per architecture, `Makefile` at the
  root, `platform/*/build-*.sh` per hypervisor target.
- **Gates** the build's own `BUILD_NUMBER` validation, and the C99/C toolchain
  checks in `AUDIT/` and the platform scripts.

## Converter — Swift, semantic version, 1 release

- **Identity** semantic version, `vX.Y`. Single release so far: `v1.0`.
- **Artifacts** a single executable `converter` plus `converter.sha256`. This is
  the minimal shape and the one to bring the other single-binary projects to.
- **To bring into line** the artifact is published as a bare binary rather than a
  named archive. A rename to `converter-X.Y-macos-arm64` with the licence and a
  `README-binaries.txt` inside would satisfy §1.6 without changing how it is used.

## MCPSearch — Swift, semantic version, 1 release

- **Identity** `vX.Y.Z`, and it is **declared in three unconnected places**: the
  authoritative-looking `static let serverVersion = "1.0.0"` in
  `Sources/SwiftWebSearchMCP/MCPServer.swift`, a second literal in the
  `clientInfo` dictionary in
  `Sources/WebSearchCore/Providers/ParallelMCPProvider.swift`, and the
  `CHANGELOG.md` heading. There is **no `VERSION` file and no check tying them
  together**, so bumping the version is manual and a half-done bump ships a server
  that misreports itself over MCP. Introducing the `VERSION` file plus an agreement
  check is the obvious next step here.
- **Artifacts** `mcps-X.Y.Z-macos-arm64.tar.gz` + `SHA256SUMS`, with the install
  instructions carried in the release notes. `v1.0.0` (2026-09-15) is the first.
- **There is no release script.** `v1.0.0` was cut by hand — no `tools/release.sh`
  exists — so the packaging, digest and notes sequence in Part 1 has to be walked
  manually and is not yet reproducible from one command.
- **`main` is unprotected** and carries no rulesets: nothing gates a merge today,
  so the checks below are advisory until that changes.
- **Code scanning uses CodeQL advanced setup** (`.github/workflows/codeql.yml`,
  `build-mode: manual`, weekly cron). **Do not switch it to default setup** — the
  runner image ships Swift 6.3.3, which cannot parse this package's 6.4 manifest, so
  default setup analyses nothing while appearing to run, and removes SAST silently.
- **AI Scan for pull requests is deliberately disabled** on this repository and
  every other non-archived one — the Autofind job asks
  `api.individual.githubcopilot.com` for a model an individual Copilot plan does not
  serve, so it failed on every PR head with `CAPIError: 400 The requested model is
  not supported` and could never report a finding. Re-enable only with an
  entitlement that serves the requested model:
  `PATCH /repos/{owner}/{repo}/code-scanning/ai-scan` with `{"pr_scan":"enabled"}`.
  The decision is recorded in `AUDIT/HANDOVER.md` as ISSUE-21.
- **Audit material** lives under `AUDIT/`; `AUDIT/HANDOVER.md` lists what is open
  — notably ISSUE-20, rotating the GitHub PAT in cleartext in the local wiki
  clones' `.git/config`.
- **Next release needs**, in order: a `VERSION` file, a release script, and the
  three version literals reduced to one.

## ChatBots — Swift, no release yet

- **Identity** semantic version, not yet established.
- **Code scanning** as for MCPSearch: CodeQL default setup, AI Scan disabled.
- Needs a release path before it can be called releasable; follow Converter's
  minimal shape once it produces a runnable target.

## TinyTitan_Datacenter — Python, no release yet

- **Identity** semantic version, not yet established.
- **No compiled artifact.** If a release is warranted it ships the harness as a
  wheel or source archive with a documented entry point, and §1.2.1–1.2.4 do not
  apply because nothing is compiled for the host.
- **Purpose** the multi-node Apple-silicon cluster: harness, measurements and
  findings. Measurements are reported as measurements, never as performance
  ceilings, and every recorded number names the commit, hardware, RAM, macOS and
  Swift/toolchain versions it was taken on.

## YTLive_Laundry — Python, no release yet

- **Identity** semantic version, not yet established.
- **Repository is public.** Its views badge uses the README-embedded static form
  rather than the endpoint form the other public repositories use; either works,
  and it is left alone rather than churning the README. Converting it means moving
  it into the `REPOS` list in `~/bin/traffic-badge-update.sh` and swapping the
  badge for the endpoint shape.
- No compiled artifact; see TinyTitan_Datacenter for the shape if a release is
  warranted.

## AISessionServer — Shell, no release yet

- **Identity** semantic version, not yet established.
- **No compiled artifact.** A release, if warranted, is a tagged source archive
  with a documented entry point, and the archive is what the Release carries.
- **Code scanning** CodeQL default setup covers `actions`.

## Minecraft — Shell, no release yet

- **Identity** semantic version, not yet established.
- **No compiled artifact**; the same source-archive shape as AISessionServer.
- Nine stale code-scanning alerts point at deleted `Server App/…` paths and want
  dismissing as no-longer-present rather than fixing.

## RoomCAD — JavaScript, no release yet

- **Identity** semantic version, not yet established.
- **No compiled artifact.** A release would ship a bundle or a package tarball;
  §1.2.1–1.2.4 are not applicable to interpreted output.

## OpenRA — C#, no release yet

**Not upstreamed. Every change stays in this repository.** Do not open pull
requests, cherry-picks or patches against `OpenRA/OpenRA`; nothing developed here
is intended for the upstream project.

This repository is a GitHub fork of `OpenRA/OpenRA` and is **deliberately left as
one** — do not detach it from the fork network. Leaving is permanent, and the
standalone repository would not retain its wiki, issues, pull requests, stars,
watchers or comments; the wiki alone holds 5 pages across 30 commits. The fork
relationship is inert: nothing reaches upstream unless someone explicitly pushes it
there, and the rule above already forbids that.

- **Identity** semantic version, not yet established.
- **Releases** none expected while this is a downstream — upstream cuts OpenRA's
  releases. If this repository ever ships artifacts of its own, they are its own
  concern and this section must say so explicitly rather than borrowing upstream's
  process.

## FXNews — MQL5, source and compiled artifact

**A release is the source file and the compiled indicator, and nothing else.**

- **Artifacts** `FXNews.mq5` — the single MetaTrader 5 source — and `FXNews.ex5`,
  the bytecode MetaEditor compiles from it. Both are attached, with a digest
  covering both, because the source is the reviewable artifact and the `.ex5` is
  the one users actually load into a terminal.
- **Identity** the `#property version "M.mpp"` line in `FXNews.mq5` — currently
  `"3.300"` for version 3.3. That is the only version literal; nothing else in the
  tree restates it, and no separate `VERSION` file is introduced. MQL5 version
  properties are two-component at most in practice, so the release tag here is
  `v3.300`, taken verbatim from the property rather than re-derived.
- **Build** `tools/build-macos.sh`, which drives MetaEditor under the
  MetaQuotes-bundled Wine on macOS. It exits non-zero on a compiler error **or
  warning**, so a warning-free compile is already enforced and the release uses the
  same script unrelaxed. Its `--install` mode copies into the live terminal folder
  for local use and is **not** part of producing a release artifact.
- **Not a macOS host binary.** §1.2.1–§1.2.4 (arm64-only, `lipo`, universal
  binaries) **do not apply**: the `.ex5` is MetaTrader 5 bytecode, and macOS's only
  involvement is hosting the Wine-based compiler. Do not attach a `lipo -archs`
  assertion to this repository — it would be a gate that cannot fail.
- **No releases yet.** If one is cut, the checklist above is the whole process; the
  generic macOS packaging sections of Part 1 do not apply here.

## FXAI, FXAI-V0 — archived, out of scope

Archived, therefore read-only: no commit, and so no release step, can touch them.
Neither carries a view badge, so nothing is left stale by excluding them. If
either is unarchived, it gets a Part 2 section before it gets a release.
