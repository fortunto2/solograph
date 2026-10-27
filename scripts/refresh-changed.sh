#!/bin/bash
# Keep the portfolio index current: add new checkouts to the registry, then
# reindex what moved, walking the registry rather than the disk; last, read new
# transcripts into the session index.
#
# Why the registry: `solograph-cli scan --project NAME` validates NAME against
# ~/.solo/registry.yaml, and the registry carries each project's path — including
# nested ones. The first version of this script looped over `~/startups/active/*/`
# instead, and got both halves wrong: video-analyzer lives one level deeper
# (life2film/video-analyzer) so it was never scanned at all — the very project
# whose six-month-stale graph prompted this script — while twelve directories
# that are not in the registry failed with "Project not found" and were counted
# as done. Walking the registry makes the mismatch impossible by construction.
#
# The registry itself is refreshed first (`solograph-cli registry`: merges, never
# replaces). It had been generated once, in February, and by October 40 projects
# in active/ were not in it, so no amount of nightly scanning reached them.
#
# Scope: nightly, the whole registry minus SOLOGRAPH_SKIP. Run by hand from
# inside a product, the script walks up looking for `.solograph.yaml` and takes
# its `scope:` list (life2film keeps one). SOLOGRAPH_SCOPE overrides both: a
# space-separated list, or "all".
#
# Run by ~/Library/LaunchAgents/com.solograph.refresh.plist, nightly; by
# `make scan` in solopreneur on demand.
set -u

SOLOGRAPH="$HOME/startups/shared/solograph"
CLI="$SOLOGRAPH/.venv/bin/solograph-cli"
REGISTRY="$HOME/.solo/registry.yaml"
STAMP_DIR="$SOLOGRAPH/.refresh-stamps"
LOG="$SOLOGRAPH/.refresh.log"
# The project's own interpreter, never `python3` from PATH: launchd's PATH puts
# Homebrew first, Homebrew moved python3 to 3.14 on 21 Aug without PyYAML, and
# from that night every run died reading the registry and logged "done —
# scanned 0, failed 0". 70 nights, zero projects scanned, nothing red.
PY="$SOLOGRAPH/.venv/bin/python"
# Projects to leave out, space-separated. rust-code used to be skipped by
# default — a deep graph scan of it took 27 minutes on 22 Aug — but a scan of
# definitions only covers all 77 projects, rust-code included, in 38 seconds.
SKIP="${SOLOGRAPH_SKIP-}"

# The log always; the terminal too when someone is watching (`make scan`).
say() {
    if [ -t 1 ]; then echo "$(date +%FT%T) $*" | tee -a "$LOG"; else echo "$(date +%FT%T) $*" >> "$LOG"; fi
}
# Walk up for a project config, the way git finds its root.
find_scope_file() {
    dir="${SOLOGRAPH_ROOT:-$PWD}"
    while [ "$dir" != "/" ] && [ "$dir" != "$HOME" ]; do
        [ -f "$dir/.solograph.yaml" ] && { echo "$dir/.solograph.yaml"; return; }
        dir=$(dirname "$dir")
    done
}

SCOPE_FILE=$(find_scope_file)
if [ -n "${SOLOGRAPH_SCOPE:-}" ]; then
    [ "$SOLOGRAPH_SCOPE" = "all" ] && WANTED="" || WANTED="$SOLOGRAPH_SCOPE"
    SCOPE_SRC="SOLOGRAPH_SCOPE"
elif [ -n "$SCOPE_FILE" ]; then
    WANTED=$("$PY" -c "
import yaml, pathlib, sys
d = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
print(' '.join(d.get('scope', [])))
" "$SCOPE_FILE" 2>>"$LOG") || { say "FAILED reading scope $SCOPE_FILE"; exit 1; }
    SCOPE_SRC="$SCOPE_FILE"
else
    WANTED=""
    SCOPE_SRC="whole registry"
fi

mkdir -p "$STAMP_DIR"

# Load average, not the hour — but only as a last resort. The job already runs
# as a launchd Background process (E-cores, low-priority I/O) plus nice 10, so
# it yields to anything interactive by itself. The old gate of 8 on a 10-core
# Mac skipped 17 of 70 nights, at loads of 16-61 from agents working at night,
# and a skipped night is a stale graph. Four runnable threads per core is a
# machine in trouble; below that, run.
limit=$(( $(sysctl -n hw.ncpu) * 4 ))
load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
if [ "$load" -gt "$limit" ]; then
    say "skipped, load $load > $limit"
    exit 0
fi

"$CLI" registry >> "$LOG" 2>&1 || { say "FAILED refreshing $REGISTRY"; exit 1; }

# Read the registry before the loop, not in a process substitution: a crash
# there is invisible to the loop, which just sees no lines and reports success.
PROJECTS=$("$PY" -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path('$REGISTRY').read_text())
for p in d.get('projects', []):
    print(f\"{p['name']}|{p['path']}\")
" 2>>"$LOG") || { say "FAILED reading $REGISTRY"; exit 1; }
[ -n "$PROJECTS" ] || { say "FAILED: $REGISTRY lists no projects"; exit 1; }
[ -n "$SKIP" ] && say "skipping: $SKIP (SOLOGRAPH_SKIP)"

scanned=0; failed=0; quiet=0
while IFS='|' read -r name path; do
    [ -n "$name" ] || continue
    if [ -n "$WANTED" ]; then
        case " $WANTED " in *" $name "*) ;; *) continue ;; esac
    fi
    case " $SKIP " in *" $name "*) continue ;; esac
    [ -d "$path/.git" ] || { say "no git checkout: $name"; continue; }

    # One stamp per project, so a failure anywhere leaves the others alone —
    # a single shared stamp marked crashed projects as fresh and they stayed
    # behind for good.
    stamp="$STAMP_DIR/$name"
    started=$(date +%Y-%m-%dT%H:%M:%S)

    # Every scan is incremental: a file is re-read only when its mtime or size
    # changed, which also catches uncommitted, reverted and untracked work that
    # a git log window cannot. A project with nothing to do says "unchanged".
    if out=$(nice -n 10 "$CLI" scan --project "$name" 2>&1); then
        echo "$out" >> "$LOG"
        echo "$started" > "$stamp"        # stamp only a pass that finished
        if echo "$out" | grep -q "$name: unchanged"; then
            quiet=$((quiet + 1))
        else
            say "scanned $name"
            scanned=$((scanned + 1))
        fi
    else
        echo "$out" >> "$LOG"
        say "FAILED $name — stamp left alone, it will retry"
        failed=$((failed + 1))
    fi
done <<< "$PROJECTS"

say "done — scanned $scanned, unchanged $quiet, failed $failed (scope: $SCOPE_SRC)"

# The session index is the archive: Claude Code deletes transcripts after 30
# days, and the index only reads them when someone searches. A month without a
# session_search would lose that month, so read new and grown transcripts here.
if sessions=$("$PY" -c "
from codegraph_mcp.session_search import SessionSearch
r = SessionSearch().sync()
print(f\"sessions — {r.get('read', 0)} read, {r.get('indexed', 0)} indexed, {r.get('gone', 0)} archived, {len(r.get('failed', []))} failed\")
" 2>>"$LOG"); then
    say "$sessions"
else
    say "FAILED syncing the session index"
    failed=$((failed + 1))
fi
[ "$failed" -eq 0 ]
