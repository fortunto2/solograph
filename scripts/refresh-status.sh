#!/bin/bash
# Is the nightly refresh actually refreshing? One line, exit 0 only when it is.
#
# Written after the refresh logged "done — scanned 0, failed 0" for 70 nights in a
# row (Aug–Oct 2026): python3 had lost PyYAML, the registry read crashed inside a
# process substitution, and every run looked like a quiet success. A log that says
# "done" is not evidence; this checks what the last run did.
#
#   OK     last run within 30 h, nothing failed, at least one project looked at
#   STALE  no finished run within 30 h (skipped on load, launchd not firing, …)
#   FAIL   the last run failed, scanned nothing at all, or the session sync failed
#
# Used by the harness check `solograph-fresh` (solopreneur/harness/checks.json).
set -u

LOG="${SOLOGRAPH_REFRESH_LOG:-$HOME/startups/shared/solograph/.refresh.log}"
MAX_AGE_H="${SOLOGRAPH_REFRESH_MAX_AGE_H:-30}"

[ -r "$LOG" ] || { echo "FAIL: no refresh log at $LOG"; exit 1; }

# Only a run over the whole registry says the nightly works: a manual run over one
# project after a failed night read as OK while 73 of 77 projects went unrefreshed
# (review, 28 Oct 2026). A FAILED anywhere after that run still fails below.
done_line=$(grep -E '^[0-9T:-]+ done — .*\(scope: whole registry\)' "$LOG" | tail -1)
[ -n "$done_line" ] || { echo "FAIL: the refresh has never finished a run over the whole registry ($LOG)"; exit 1; }

stamp=${done_line%% *}
then_s=$(date -j -f %Y-%m-%dT%H:%M:%S "$stamp" +%s 2>/dev/null) ||
    { echo "FAIL: unreadable timestamp in: $done_line"; exit 1; }
age_h=$(( ($(date +%s) - then_s) / 3600 ))

read -r scanned quiet failed < <(echo "$done_line" |
    sed -E 's/.*scanned ([0-9]+), unchanged ([0-9]+), failed ([0-9]+).*/\1 \2 \3/')
after_done=$(sed -n "/^$(printf '%s' "$stamp" | sed 's/[]\/$*.^[]/\\&/g') done — /,\$p" "$LOG")
sessions=$(echo "$after_done" | grep -E ' sessions — ' | tail -1)

if [ "$age_h" -ge "$MAX_AGE_H" ]; then
    echo "STALE: last finished run ${age_h} h ago ($stamp); latest: $(tail -1 "$LOG")"
    exit 1
fi
if [ "${failed:-1}" -ne 0 ] || echo "$after_done" | grep -q 'FAILED'; then
    echo "FAIL: $done_line$(echo "$after_done" | grep -m1 FAILED | sed 's/^/ | /')"
    exit 1
fi
if [ "$((scanned + quiet))" -eq 0 ]; then
    echo "FAIL: the last run looked at no project at all — the 70-night failure mode ($stamp)"
    exit 1
fi
if [ -z "$sessions" ] || ! echo "$sessions" | grep -q ', 0 failed'; then
    echo "FAIL: session sync missing or failed after $stamp: ${sessions:-no sessions line}"
    exit 1
fi
echo "OK: ${age_h} h ago — scanned $scanned, unchanged $quiet; ${sessions#* sessions — }"
