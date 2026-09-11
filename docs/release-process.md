# Cutting an NVMAI release

This is the runbook for turning a green `main` into a tagged, published release
with prebuilt binaries. It exists because the sequence has traps that cost time
and one that can publish the wrong thing. `CONTRIBUTING.md` has the short
version; this file is the one to follow, including what to do when it stalls.

Every release is a **tag and a published GitHub Release with binaries**. The
mechanism is `tools/release.sh`, the shape is the previous release, and the
release notes are `docs/release-notes-vX.Y.md`.

## 0. What you need before starting

| Check | Command | Why |
| --- | --- | --- |
| macOS 26+, Swift 6.3+ | `sw_vers`, `swift --version` | The runtime's floor; the release notes state it |
| Disk | `df -h .` | A clean scratch build plus the staged archive wants ~10 GB |
| Memory | `memory_pressure -Q` | The golden baselines load real models |
| **No model process** | `pgrep -fl 'NVMAIServer\|NVMAIMac\|NVMAIDecodeService\|NVMAICLI\|NVMAIPackageTests\|swiftpm-testing-helper\|mlx_lm\|mlx-lm'` | The golden gate refuses to run beside one; see §5 |
| `gh` authenticated | `gh auth status` | Publishing uses it; it must be the repo owner's account |
| Clean tree, HEAD on the tag | `git status --porcelain` | `release.sh` enforces both |

Never terminate a process you did not start. If one of these is alive and not
yours, stop and ask the human — §5 is the long version.

## 1. Prepare the version

Three places, and only the first is a literal:

1. **`tools/install_nvmai.sh`** — `CFBundleVersion` and
   `CFBundleShortVersionString` in the app bundle it writes. That is the only
   version literal in the tree. Grep for the previous version before believing
   this: `grep -rn "5\.1\b" --include="*.sh" --include="*.swift" sources/ tools/`.
2. **The wiki `Changelog.md`** (`.qwen/wiki/Changelog.md`) — a new `## X.Y — <headline>`
   section at the top, with `[Release vX.Y](https://github.com/Pummelchen/NVMAI/releases/tag/vX.Y)`
   and user-facing bullets. Keep it compact: what a *user* can do now that they
   could not before, and the numbers that back it.
3. **`README.md`** — the `## New in X.Y` callout immediately after the intro
   line, replacing the previous one. This is the release process's step 1 and
   the first thing a visitor reads.

The dated `docs/site/*.md` articles say "at the time of writing" and are **not**
bumped: they record when they were verified, and re-stamping them without
re-verifying would be a false claim.

## 2. Write the release notes

`docs/release-notes-vX.Y.md`, modelled on the previous one:

- a `## NVMAI X.Y — <headline>` title, then one paragraph saying what the
  release is for;
- one `###` section per user-visible change, each naming the check that backs
  it (a gate, a measurement, a real-model run);
- `### Also in this release` for the smaller items;
- `### Performance` — only numbers measured on *this* commit, and say plainly
  when a previous table was not re-run;
- a final section:

  ```
  ### Checksum

  `nvmai-X.Y-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
  ```

`SHA256_PENDING` is not a placeholder to forget: `release.sh --publish`
substitutes it with the archive it just built, and **refuses to publish if the
notes neither carry it nor quote the real digest**. A release whose notes quote
the wrong digest is worse than one quoting none — 3.7 shipped that way for a
few minutes.

The last two sections are a claim about what was verified. Do not write a gate
result you have not seen; add it after the dry run if you want it in the notes.

## 3. Commit, tag, push

The annotated tag's message is the starting point for the notes, so make it the
headline plus the lead paragraph.

```bash
git add -A && git commit -m "Prepare X.Y: release notes, the README callout, and the app version"
git push origin main
git tag -a vX.Y -m "NVMAI X.Y — <headline>

<lead paragraph>"
git push origin vX.Y
```

Push the wiki's Changelog too (`git -C .qwen/wiki commit && git -C .qwen/wiki push`).
`release.sh` requires HEAD to *be* the tag and the tag to be *pushed*; a tag
that exists only locally fails the precondition with a clear message.

## 4. Dry run, then publish

```bash
tools/release.sh vX.Y                     # verify, build, stage — no publish
tools/release.sh vX.Y --publish --notes docs/release-notes-vX.Y.md
```

The dry run is the default because publishing notifies watchers. Read its
output; then look at `.build/releases/nvmai-release-X.Y/` — the staged tree and
the tarball — before re-running with `--publish`.

What the dry run does, in order:

1. **Preconditions** — clean tree, `vX.Y` exists locally, HEAD is that commit,
   the tag is on `origin`, and no Release for it exists yet.
2. **Gates** — `tools/lint.sh`; `swift test --no-parallel`, which must print
   `Test run with N tests in M suites passed`; then **every installed model that
   has a golden target** (`benchmark/golden/`), each through
   `tools/golden-baseline.sh --check <target>`.
3. **A clean scratch build** — `swift build -c release --scratch-path
   .build/releases/.../build`, with the log scanned for compiler warnings. It is
   deliberately not an incremental build: an incremental one compiles nothing
   and the warning gate passes vacuously.
4. **Stage and package** — the six executables, the `.bundle` resources (the
   Metal shader library — the runtime cannot load kernels without them), the
   licence and notices, `README-binaries.txt`, then the tarball and its
   `.sha256`.

`release.sh` requires HEAD to *be* the tag, so if a commit landed on `main`
after tagging (a documentation fix is the usual reason), run the release from
the tag itself — the binaries are built from the tagged commit either way:

```bash
git checkout vX.Y          # detached HEAD on the tag
tools/release.sh vX.Y      # then --publish
git checkout main
```

`--publish` repeats all of that and then creates the Release with `--repo`
pinned. Every `gh` call is pinned because in a fork `gh` defaults to the
*parent* repository: `gh release list` would show another project's releases and
`gh release create` fails with a misleading "tag has not been pushed".

## 5. When a golden gate refuses (the trap that looks like a failure)

If the log shows this, **no golden was compared**:

```
refusing to start: these processes match the model-process guard
  91687 .../swiftpm-testing-helper --test-bundle-path .../WebTransport...
stop them yourself, or re-run when they are gone. This script never terminates a process it did not start.
error: golden baseline mismatch (qwen38-4)
```

`release.sh` reports the refusal as `golden baseline mismatch (<target>)`,
because its helper exits non-zero for both. Check the line above it: a real
mismatch prints an output diff instead. Do not re-capture a baseline to make
this go away — the baseline is valid for one (machine, build, model) triple.

The common blocker on a shared machine is another project's
`swiftpm-testing-helper` (a Dropbox-resident checkout, in this workspace). It
can be a *loop* that respawns every couple of minutes; the guard is re-checked
by **each** of the eight golden invocations, so a loop with a 40% duty cycle
means the phase cannot complete. What to do:

1. Tell the human which process is blocking, with its parent and how long it
   has been alive (`ps -o pid,ppid,etime,%cpu -p <pid>`).
2. Ask them to stop it, or wait for it to finish. Never terminate it yourself,
   and never `pkill` by pattern.
3. When the machine has been quiet for ~90 s, re-run the dry run.

Waiting for a *single* short gap is not enough: the gate must start cleanly
eight times. Verify a real quiet window before spending another attempt:

```bash
for i in $(seq 1 6); do pgrep -f 'swiftpm-testing-help[e]r' >/dev/null && echo busy || echo quiet; sleep 30; done
```

(Bracket the pattern — `help[e]r` — or `pgrep` matches the shell running it.)

## 6. After publishing

```bash
gh release view vX.Y --repo Pummelchen/NVMAI --json url,assets \
  --jq '"\(.url) \([.assets[].name] | join(", "))"'
```

Check: the notes on the Release quote the digest in the archive's `.sha256`
next to it; the assets are the tarball and the checksum; the wiki Changelog
points at the same tag. The binaries are **not** signed or notarized, and
`README-binaries.txt` in the archive says so and tells the user how to clear the
quarantine attribute after verifying the checksum — keep that honest rather
than implying a notarized build.

If you wrote a `### Performance` table, make sure it says which commit and
machine it was measured on, and leave previous releases' tables alone.

## 7. Checklist

- [ ] Installer version bumped; no stale version literal left (`grep` for it)
- [ ] Wiki `Changelog.md` has the new section, pushed
- [ ] README has `## New in X.Y`, replacing the previous callout
- [ ] `docs/release-notes-vX.Y.md` ends with a `SHA256_PENDING` checksum block
- [ ] Tree clean, `git tag -a vX.Y`, tag pushed, `release.sh` preconditions pass
- [ ] Dry run green: lint, the serial suite, **every installed golden**, a
      warning-free clean build
- [ ] Staged archive inspected (six executables, bundles, licence, notices)
- [ ] `--publish --notes docs/release-notes-vX.Y.md`, then `gh release view`
- [ ] No model process left running afterwards
