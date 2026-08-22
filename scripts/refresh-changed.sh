#!/bin/bash
# Reindex the projects that moved, walking the registry rather than the disk.
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
# Scope: which projects belong together is a fact about the product, not about
# this tool, so it lives with the product. The script walks up from where it is
# run (or from SOLOGRAPH_ROOT) looking for `.solograph.yaml` and takes its
# `scope:` list. Without one it scans every registry entry. SOLOGRAPH_SCOPE
# overrides both: a space-separated list, or "all".
#
# Run by ~/Library/LaunchAgents/com.solograph.refresh.plist, nightly.
set -u

SOLOGRAPH="$HOME/startups/shared/solograph"
REGISTRY="$HOME/.solo/registry.yaml"
STAMP_DIR="$SOLOGRAPH/.refresh-stamps"
LOG="$SOLOGRAPH/.refresh.log"
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
    WANTED=$(python3 -c "
import yaml, pathlib, sys
d = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
print(' '.join(d.get('scope', [])))
" "$SCOPE_FILE")
    SCOPE_SRC="$SCOPE_FILE"
else
    WANTED=""
    SCOPE_SRC="whole registry"
fi

mkdir -p "$STAMP_DIR"
say() { echo "$(date +%FT%T) $*" >> "$LOG"; }

# Load average, not the hour: the machine is busy at night too when agents run,
# and an unattended reindex must never be the reason a build feels slow.
load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
if [ "$load" -gt 8 ]; then
    say "skipped, load $load"
    exit 0
fi

scanned=0; failed=0; quiet=0
while IFS='|' read -r name path; do
    [ -n "$name" ] || continue
    if [ -n "$WANTED" ]; then
        case " $WANTED " in *" $name "*) ;; *) continue ;; esac
    fi
    [ -d "$path/.git" ] || { say "no git checkout: $name"; continue; }

    # One stamp per project, so a failure anywhere leaves the others alone —
    # a single shared stamp marked crashed projects as fresh and they stayed
    # behind for good.
    stamp="$STAMP_DIR/$name"
    since=$(cat "$stamp" 2>/dev/null || date -v-7d +%Y-%m-%dT%H:%M:%S)
    started=$(date +%Y-%m-%dT%H:%M:%S)

    moved=$(git -C "$path" log --since="$since" --oneline 2>/dev/null | head -1)
    [ -n "$moved" ] || { quiet=$((quiet + 1)); continue; }

    say "scanning $name"
    if nice -n 10 "$SOLOGRAPH/.venv/bin/solograph-cli" scan --project "$name" --deep >> "$LOG" 2>&1; then
        echo "$started" > "$stamp"        # stamp only a pass that finished
        scanned=$((scanned + 1))
    else
        say "FAILED $name — stamp left alone, it will retry"
        failed=$((failed + 1))
    fi
done < <(python3 -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path('$REGISTRY').read_text())
for p in d.get('projects', []):
    print(f\"{p['name']}|{p['path']}\")
")

say "done — scanned $scanned, unchanged $quiet, failed $failed (scope: $SCOPE_SRC)"
[ "$failed" -eq 0 ]
