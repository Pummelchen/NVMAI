#!/usr/bin/env bash
# Production gates that the compiler cannot express. Run locally before a PR;
# CI runs the same script, so a green run here is a green run there.
#
#   tools/lint.sh              # all checks
#   tools/lint.sh force-cast   # one check
#
# Checks:
#   force-cast   no `as!` / `try!` in sources/ without an audited opt-out
#   func-length  no NEW function longer than MAX_FUNC_LINES (ratcheted)
#
# Opting out of force-cast: put `lint:allow-force <reason>` in a comment on
# the line immediately above. The reason is mandatory and is what a reviewer
# reads — an opt-out without one fails the same as no opt-out at all.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="$SCRIPT_DIR/func-length-baseline.txt"
MAX_FUNC_LINES="${MAX_FUNC_LINES:-120}"

status=0
want="${1:-all}"

# --- force-cast / force-try -------------------------------------------------
check_force_cast() {
  echo "== force-cast: as! / try! outside tests =="
  local found=0
  while IFS= read -r hit; do
    local file line
    file="${hit%%:*}"
    line="$(echo "$hit" | cut -d: -f2)"
    # Walk up the contiguous comment block directly above the hit, looking for
    # an opt-out marker followed by a reason. Scanning the whole block (not
    # just the previous line) lets the reason wrap naturally.
    local n=$((line - 1)) text ok=0
    while [ "$n" -ge 1 ]; do
      text="$(sed -n "${n}p" "$file")"
      echo "$text" | grep -qE '^[[:space:]]*//' || break
      if echo "$text" | grep -qE 'lint:allow-force[[:space:]]+[^[:space:]]'; then
        ok=1
        break
      fi
      n=$((n - 1))
    done
    [ "$ok" -eq 1 ] && continue
    echo "  ${file#$ROOT/}:$line: $(echo "$hit" | cut -d: -f3- | sed 's/^[[:space:]]*//')"
    found=1
  done < <(grep -rnE '(\bas!\s|\btry!\s)' --include='*.swift' "$ROOT/sources" 2>/dev/null)

  if [ "$found" -ne 0 ]; then
    echo "  FAIL: force cast/try without an audited 'lint:allow-force <reason>' comment above it"
    status=1
  else
    echo "  ok"
  fi
}

# --- function length --------------------------------------------------------
# Indentation-anchored: a function runs from its `func` line to the first line
# that closes a brace at the same indent. Brace-depth counting drifts on braces
# inside strings and comments; this codebase is consistently formatted, so
# indent is the more reliable anchor.
#
# Every `func` must land in exactly one of three buckets: no body (a protocol
# requirement), a body that opens and closes on one line, or a body with a
# closer at its own indent. Anything else is printed as UNRESOLVED and fails
# the check. A gate that silently skips what it cannot parse reports "ok" for
# code it never looked at, which is worse than no gate — so unparsed input is
# an error, not a shrug.
measure_functions() {
  ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    limit = Integer(ENV.fetch("MAX_FUNC_LINES", "120"))
    root = ENV.fetch("ROOT")
    scanned = 0
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      lines = File.readlines(path, chomp: true)
      rel = path.delete_prefix(root + "/")
      lines.each_with_index do |line, i|
        next unless (m = line.match(/^(\s*)(?:[\w@\(\)]+\s+)*func\s+([A-Za-z_]\w*)/))
        scanned += 1
        indent, name = m[1], m[2]

        # Walk the (possibly multi-line) signature looking for the body brace.
        # Stop at the next declaration or at a closer no deeper than us, which
        # is what a bodyless protocol requirement runs into.
        open_at = nil
        j = i
        while j < lines.length
          text = lines[j].sub(%r{//.*$}, "")
          if j > i && text =~ /^\s{0,#{indent.length}}(\}|func\s|var\s|let\s|case\s)/
            break
          end
          if text.include?("{")
            open_at = j
            break
          end
          j += 1
        end

        if open_at.nil?
          next # no body: protocol requirement or bodyless declaration
        end

        opener = lines[open_at].sub(%r{//.*$}, "")
        if opener.count("{") == opener.count("}") && opener.rstrip.end_with?("}")
          next # body opens and closes on one line
        end

        closer = /^#{indent}\}/
        stop = ((open_at + 1)...lines.length).find { |k| lines[k] =~ closer }
        if stop.nil?
          puts "UNRESOLVED:#{rel}:#{name}:#{i + 1}"
          next
        end
        length = stop - i
        next unless length > limit
        puts "#{rel}:#{name}:#{length}"
      end
    end
    # Coverage receipt. Without it an empty result is indistinguishable from
    # "scanner never ran", and the gate would report ok for an unexamined tree.
    puts "SCANNED:#{scanned}"
  '
}

check_func_length() {
  echo "== func-length: no NEW function over $MAX_FUNC_LINES lines =="
  local measured current new unresolved raw rc scanned

  # Capture without a pipe so the scanner's exit status survives, then check it.
  # A gate whose measurement step died must fail, not report "ok (0 new)" —
  # that is how an unexported ROOT once let this check pass while looking at
  # nothing at all.
  raw="$(measure_functions)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the length scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the length scanner reported no functions scanned."
    echo "        Expected ~1000 under sources/; check ROOT and the glob."
    status=1
    return
  fi
  measured="$(echo "$raw" | grep -v '^SCANNED:' | sort)"

  # Coverage first: if the scanner could not resolve a function, the ratchet
  # below is reporting on an unknown subset of the tree. Fail loudly rather
  # than let an "ok" stand for code that was never measured.
  unresolved="$(echo "$measured" | grep '^UNRESOLVED:' || true)"
  if [ -n "$unresolved" ]; then
    echo "$unresolved" | sed 's/^UNRESOLVED:/  UNRESOLVED: /'
    echo "  FAIL: the length scanner could not find these functions' bounds."
    echo "        Fix tools/lint.sh — do not silence this by ignoring them."
    status=1
    return
  fi
  current="$(echo "$measured" | grep -v '^UNRESOLVED:' || true)"

  if [ ! -f "$BASELINE" ]; then
    echo "  no baseline at ${BASELINE#$ROOT/}; writing one"
    echo "$current" > "$BASELINE"
    echo "  ok (baseline created, $(echo "$current" | grep -c . ) entries)"
    return
  fi
  # Compare on file:function only — a baselined function is allowed to exist,
  # and Phase 2 shrinking it should not require a baseline edit.
  new="$(comm -13 <(cut -d: -f1,2 "$BASELINE" | sort -u) \
                  <(echo "$current" | cut -d: -f1,2 | sort -u))"
  if [ -n "$new" ]; then
    echo "$new" | sed 's/^/  NEW: /'
    echo "  FAIL: shorten it, or update ${BASELINE#$ROOT/} with a reason in the PR"
    status=1
  else
    echo "  ok ($(echo "$current" | grep -c .) baselined, 0 new, $scanned scanned)"
  fi
}

case "$want" in
  all)         check_force_cast; check_func_length ;;
  force-cast)  check_force_cast ;;
  func-length) check_func_length ;;
  *) echo "unknown check: $want (all|force-cast|func-length)" >&2; exit 2 ;;
esac

exit $status
