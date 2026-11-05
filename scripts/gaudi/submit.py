"""Submit an API workflow and save its history, failing on execution errors."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("workflow", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8189")
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
prompt = json.loads(args.workflow.read_text())
request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt}).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    submitted = json.load(response)
prompt_id = submitted["prompt_id"]
print("prompt_id:", prompt_id, flush=True)
start = time.monotonic()
while True:
    with urllib.request.urlopen(args.url + "/history/" + prompt_id) as response:
        history = json.load(response)
    if prompt_id in history:
        result = history[prompt_id]
        result["wall_seconds"] = time.monotonic() - start
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2))
        print(json.dumps({"status": result["status"], "outputs": result["outputs"], "wall_seconds": result["wall_seconds"]}), flush=True)
        if result["status"]["status_str"] != "success":
            raise SystemExit(1)
        break
    time.sleep(5)
