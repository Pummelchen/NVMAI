# The Swift language standard

What this tree is written to, what the compiler enforces, and what was
deliberately left for its own pass. Measured on 2026-09-11 with the toolchain
below; the numbers come from builds, not from reading release notes.

## The baseline

| | |
| --- | --- |
| Toolchain | Apple Swift **6.3.3** (`swiftlang-6.3.3.1.3`), macOS 26.6.2, arm64 |
| Manifest | `// swift-tools-version: 6.3` |
| Language mode | `swiftLanguageModes: [.v6]` -- Swift 6 mode, so strict concurrency and the graduated 6.0/6.1/6.2 features are already errors, not warnings |

Features that are no longer "upcoming" -- they graduated into language mode 6
and are therefore already enforced -- were confirmed against the compiler by
probing `-enable-upcoming-feature <name>` and finding the flag retired:
`StrictConcurrency`, `ConciseMagicFile`, `ForwardTrailingClosures`,
`BareSlashRegexLiterals`, `IsolatedDefaultValues`, `DisableOutwardActorInference`,
`GlobalActorIsolatedTypesUsability`, `InferSendableFromCaptures`.

## Enforced: the three that are still upcoming and the tree is clean under

Declared once in `Package.swift` as `nvmaiLanguageStandard` and applied to
**every target** (28 of 28), so a target added later cannot quietly opt out:

```swift
.enableUpcomingFeature("InferIsolatedConformances")
.enableUpcomingFeature("ImmutableWeakCaptures")
.enableUpcomingFeature("MemberImportVisibility")
```

`MemberImportVisibility` was the only one that cost anything: **12 files** used
members of a module they did not import and now name it directly (`NVMAI` in
five, `NIOCore` in four, `Metal` in two, `Tokenizers` in two). That is the
feature doing its job -- a member reached through a transitive import is a
dependency the file never declared. The first measurement of this was 13
diagnostics; the compiler stops at the first error per file, so the honest count
came from iterating builds until clean, and the loop is what found the rest.

Verified with the flags coming from the manifest (no `-Xswiftc`): build 0
errors / 0 warnings, `tools/lint.sh` clean, **1458 tests in 226 suites** pass,
and the golden baseline is byte-identical on Ornith 4-bit.

## Deliberately not adopted, with the measured cost

| Feature | Cost measured | Why not now |
| --- | --- | --- |
| `ExistentialAny` | 63 errors, **32,332 warnings** | Requiring `any` at every existential position is a sweep of that size across the tree. The public API already spells `any` in most positions (hence the small error count); the bulk is internal. It is a mechanical, reviewable pass, but a large one -- its own task with the full gates, not a flag flip. |
| `InternalImportsByDefault` | **918 errors** | Changes what every `import` in every target means, and the fix is a `public import` at each re-export. Worth doing for the same reason `MemberImportVisibility` was (fewer accidental re-exports), but 918 sites is its own pass. |
| `NonisolatedNonsendingByDefault` | 79 errors **and** a semantics change | "Approachable concurrency" makes a `nonisolated async` function run on the *caller's* actor. That moves work between executors, so it needs a dedicated pass over the concurrency-sensitive paths -- the area this project's audit treated most carefully -- plus a fresh golden baseline, not a flag. |

## Small adoptions still open

Found while measuring, each mechanical and behaviour-preserving:

- `Task.sleep(nanoseconds:)` -> `Task.sleep(for:)` with `Duration`: **15 sites**
  (1 in `sources/`, the rest in tests). The `Duration` form is the modern one and
  reads better (`for: .milliseconds(5)`).
- `NSLock` -> `Synchronization.Mutex`: **28 sites**. Not a blanket change and not
  a deprecation: `Mutex`'s advantage is binding the state it protects into the
  mutex (`Mutex<State>`), which is a per-site design improvement rather than a
  rename. New code should use `Mutex` and bind the state; existing locks migrate
  when the code around them is touched for another reason.

## For new code

- `any` at every existential position, `some` where the concrete type is
  enough; the compiler will require it here once the sweep above lands.
- Typed throws (`throws(SomeError)`) where a function's failure set is closed
  and callers switch on it; untyped `throws` elsewhere.
- `Synchronization.Mutex` for new mutual exclusion, with the protected state
  inside it.
- `Duration`-based `Task.sleep(for:)` and timeouts.
- `nonisolated(unsafe)` only with the invariant written directly above it, in
  the `unchecked-invariant:` form the lint gate checks.
- A new file's entry point: `main.swift` only for top-level code; an `@main`
  type lives in a file named after it (see `docs/repository-layout.md`).
