#!/bin/bash
# Pull the YouTube spool that scripts/yt-spool.py fills on another machine, then index it here.
#
#   YT_SPOOL_HOST=<ssh host> scripts/yt-pull.sh
#
# 1. Push this machine's channels.yaml and the ids already indexed (known.txt) to the spool,
#    so the fetcher looks at the same channels and skips what is here.
# 2. Move finished <id>.info.json + <id>.vtt pairs into ~/.solo/sources/youtube/vtt/.
# 3. `solograph-cli index-youtube -u` on those ids: both files are cached, YouTube is not asked.
# A sleeping or offline machine just catches up on the next run; nothing is lost on the spool side.
set -euo pipefail
HOST=${YT_SPOOL_HOST:?set YT_SPOOL_HOST to the ssh host that runs yt-spool.py}
REMOTE=${YT_SPOOL_DIR:-.solo/yt-spool}
YT=$HOME/.solo/sources/youtube
VTT=$YT/vtt
DB=$HOME/.solo/sources/search.db
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
mkdir -p "$VTT"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/yt-pull.XXXXXX")
trap 'rm -f "$TMP"/*; rmdir "$TMP"' EXIT

if ! "${SSH[@]}" "$HOST" "mkdir -p $REMOTE"; then
  # No network (a plane, a dead hotspot) is not a failure, the next tick catches up.
  # Network up and the host silent is: the job goes red and jobs-status says so.
  if ! curl -sI -o /dev/null -m 8 https://www.cloudflare.com/; then
    echo "$(date '+%F %T') yt-pull: offline, next tick"
    exit 0
  fi
  # A paused mesh VPN looks exactly like a dead host (9 Nov 2026: the box was up, WARP was paused).
  if command -v warp-cli >/dev/null && ! warp-cli status 2>/dev/null | grep -q "Connected"; then
    echo "$(date '+%F %T') yt-pull: $HOST unreachable and WARP is not connected; if the host is reached over WARP, that is why" >&2
    exit 1
  fi
  echo "$(date '+%F %T') yt-pull: $HOST unreachable while the network is up" >&2
  exit 1
fi
sqlite3 "$DB" "SELECT video_id FROM videos" > "$TMP/known.txt"
rsync -e "${SSH[*]}" -q "$(readlink -f "$YT/channels.yaml")" "$HOST:$REMOTE/channels.yaml"
rsync -e "${SSH[*]}" -q "$TMP/known.txt" "$HOST:$REMOTE/known.txt"

# Only pairs: the fetcher writes .info.json first and .vtt last, so a .vtt means both are there.
"${SSH[@]}" "$HOST" "cd $REMOTE && for v in *.vtt; do [ -e \"\$v\" ] && [ -e \"\${v%.vtt}.info.json\" ] && echo \"\${v%.vtt}\"; done; true" > "$TMP/ready.txt"
if [ ! -s "$TMP/ready.txt" ]; then
  echo "$(date '+%F %T') yt-pull: nothing new on $HOST"
  exit 0
fi
sed -e 's/$/.info.json/' "$TMP/ready.txt" > "$TMP/files.txt"
sed -e 's/$/.vtt/' "$TMP/ready.txt" >> "$TMP/files.txt"
rsync -e "${SSH[*]}" -q --remove-source-files --files-from="$TMP/files.txt" "$HOST:$REMOTE/" "$VTT/"

n=$(wc -l < "$TMP/ready.txt" | tr -d ' ')
echo "$(date '+%F %T') yt-pull: $n videos from $HOST, indexing"
args=()
while read -r id; do args+=(-u "https://www.youtube.com/watch?v=$id"); done < "$TMP/ready.txt"
solograph-cli index-youtube "${args[@]}" | grep -E "Indexed|Already|No transcript|Error|error" || true
