#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(pwd -P)

mkdir -p .run
exec 200>.run/hpu.lock
flock -x 200

is_server() {
    [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null &&
        [[ "$(readlink "/proc/$1/cwd")" == "$root" ]] &&
        [[ "$(tr '\0' '\n' 2>/dev/null < "/proc/$1/cmdline" | sed -n '2p')" == main.py ]]
}

if [[ ! -f .run/hpu.pid ]]; then
    echo "ComfyUI HPU is not running under this script."
    exit 0
fi
read -r pid < .run/hpu.pid
if ! is_server "$pid"; then
    rm -f .run/hpu.pid
    echo "Removed stale PID file; no process was stopped."
    exit 0
fi

kill -TERM "$pid"
for ((i=0; i<60; i++)); do
    if ! is_server "$pid"; then
        rm -f .run/hpu.pid
        echo "ComfyUI HPU stopped (PID $pid)."
        exit 0
    fi
    sleep 1
done

echo "PID $pid is still shutting down. Check $root/logs/hpu.log and retry." >&2
exit 1
