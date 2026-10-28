#!/bin/bash
# Keep the portfolio index current: add new checkouts to the registry, then
# reindex what moved, walking the registry rather than the disk; last, read new
# transcripts into the session index.
#
# Why the registry: `solograph scan --project NAME` validates NAME against
# ~/.solo/registry.yaml, and the registry carries each project's path — including
# nested ones. The first version of this script looped over `~/startups/active/*/`
# instead, and got both halves wrong: video-analyzer lives one level deeper
# (life2film/video-analyzer) so it was never scanned at all — the very project
# whose six-month-stale graph prompted this script — while twelve directories
# that are not in the registry failed with "Project not found" and were counted
# as done. Walking the registry makes the mismatch impossible by construction.
#
# The registry itself is refreshed first (`solograph registry`: merges, never
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
# The Rust binary (rust/, `make rust-install`) — since 0.9 the only implementation
# of scan, registry and session sync; Python keeps only the scrapers. Its output
# keeps the wording this script and refresh-status.sh count ("NAME: unchanged",
# "sessions — N read, …").
CLI="$HOME/.solo/bin/solograph"
REGISTRY="$HOME/.solo/registry.yaml"
STAMP_DIR="$SOLOGRAPH/.refresh-stamps"
LOG="$SOLOGRAPH/.refresh.log"
# YAML is read by the binary, never by a Python: Homebrew moved python3 to 3.14
# on 21 Aug without PyYAML and 70 nights scanned zero projects; on 28 Oct the
# project venv was being rebuilt at 04:10 and the run failed. `registry --list`
# parses with the same code the scan does.
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
        # The same file can hold only the KB's settings (`kb:`); a scope needs `scope:`.
        grep -qs '^scope:' "$dir/.solograph.yaml" && { echo "$dir/.solograph.yaml"; return; }
        dir=$(dirname "$dir")
    done
}

# Which projects: the binary narrows the listing, and fails on a name the registry
# lacks — a typo used to scan nothing and say nothing.
SCOPE_FILE=$(find_scope_file)
NARROW=""; NARROW_TO=""
if [ -n "${SOLOGRAPH_SCOPE:-}" ]; then
    if [ "$SOLOGRAPH_SCOPE" = "all" ]; then
        SCOPE_SRC="whole registry"      # refresh-status counts only these runs
    else
        NARROW="--only"; NARROW_TO="$SOLOGRAPH_SCOPE"; SCOPE_SRC="SOLOGRAPH_SCOPE"
    fi
elif [ -n "$SCOPE_FILE" ]; then
    NARROW="--scope"; NARROW_TO="$SCOPE_FILE"
    SCOPE_SRC="$SCOPE_FILE"
else
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
PROJECTS=$("$CLI" registry --list --registry "$REGISTRY" ${NARROW:+$NARROW "$NARROW_TO"} 2>>"$LOG") \
    || { say "FAILED reading $REGISTRY${NARROW:+ $NARROW $NARROW_TO}"; exit 1; }
[ -n "$PROJECTS" ] || { say "FAILED: $REGISTRY lists no projects"; exit 1; }
[ -n "$SKIP" ] && say "skipping: $SKIP (SOLOGRAPH_SKIP)"

scanned=0; failed=0; quiet=0
while IFS='|' read -r name path; do
    [ -n "$name" ] || continue
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
sync_rc=0
sessions=$("$CLI" sync-sessions 2>>"$LOG") || sync_rc=$?
if [ "$sync_rc" -eq 0 ]; then
    say "$sessions"
else
    say "FAILED syncing the session index"
    failed=$((failed + 1))
fi
[ "$failed" -eq 0 ]
