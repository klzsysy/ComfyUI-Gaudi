#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(pwd -P)

if [[ -f .gaudi.env ]]; then
    source .gaudi.env
fi
listen=${COMFYUI_LISTEN:-127.0.0.1}
port=${COMFYUI_PORT:-8189}
probe_host=$listen
[[ "$probe_host" != 0.0.0.0 ]] || probe_host=127.0.0.1

mkdir -p .run logs
exec 200>.run/hpu.lock
flock -x 200

is_server() {
    [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null &&
        [[ "$(readlink "/proc/$1/cwd")" == "$root" ]] &&
        [[ "$(tr '\0' '\n' 2>/dev/null < "/proc/$1/cmdline" | sed -n '2p')" == main.py ]]
}

if [[ -f .run/hpu.pid ]]; then
    read -r pid < .run/hpu.pid
    if is_server "$pid"; then
        echo "ComfyUI HPU is already running (PID $pid)."
        exit 0
    fi
    rm -f .run/hpu.pid
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python in $root" >&2
    exit 1
fi

.venv/bin/python - "$listen" "$port" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY

printf '\n[%s] Starting ComfyUI HPU\n' "$(date -Is)" >> logs/hpu.log
nohup bash scripts/gaudi/start.sh --listen "$listen" --port "$port" \
    >> logs/hpu.log 2>&1 < /dev/null 200>&- &
pid=$!
echo "$pid" > .run/hpu.pid

for ((i=0; i<120; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f .run/hpu.pid
        echo "ComfyUI HPU exited during startup. Last log lines:" >&2
        tail -n 30 logs/hpu.log >&2
        exit 1
    fi
    if is_server "$pid" && curl --noproxy '*' --fail --silent --max-time 2 \
        "http://$probe_host:$port/system_stats" > /dev/null; then
        echo "ComfyUI HPU is ready: http://$probe_host:$port (PID $pid)"
        echo "Log: $root/logs/hpu.log"
        exit 0
    fi
    sleep 1
done

echo "Startup is still pending (PID $pid). Check $root/logs/hpu.log" >&2
exit 1
