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
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
# indent is the more reliable anchor. A function whose closer is never found is
# skipped rather than guessed at, so the ratchet never invents a violation.
measure_functions() {
  ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    limit = Integer(ENV.fetch("MAX_FUNC_LINES", "120"))
    root = ENV.fetch("ROOT")
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      lines = File.readlines(path, chomp: true)
      lines.each_with_index do |line, i|
        next unless (m = line.match(/^(\s*)(?:[\w@\(\)]+\s+)*func\s+([A-Za-z_]\w*)/))
        indent, name = m[1], m[2]
        closer = /^#{indent}\}/
        stop = nil
        ((i + 1)...lines.length).each do |j|
          if lines[j] =~ closer
            stop = j
            break
          end
        end
        next if stop.nil?
        length = stop - i
        next unless length > limit
        puts "#{path.delete_prefix(root + "/")}:#{name}:#{length}"
      end
    end
  '
}

check_func_length() {
  echo "== func-length: no NEW function over $MAX_FUNC_LINES lines =="
  local current new
  current="$(measure_functions | sort)"
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
    echo "  ok ($(echo "$current" | grep -c .) baselined, 0 new)"
  fi
}

case "$want" in
  all)         check_force_cast; check_func_length ;;
  force-cast)  check_force_cast ;;
  func-length) check_func_length ;;
  *) echo "unknown check: $want (all|force-cast|func-length)" >&2; exit 2 ;;
esac

exit $status
