# MiniMax-H3-Image job runner. `h3 gen` / `h3 edit` run this inside the Colab
# VM's kernel (via `colab exec`), prepending:
#   H3_JOBS  list of dicts: {"kind": "t2i"|"edit", "prompt": ..., ...H3Image kwargs}
# Prints one `H3_RESULT {json}` line per job (the local CLI downloads from it)
# and `H3_DONE {}` at the end.

import json
import os
import subprocess
import sys

if "/content" not in sys.path:
    sys.path.insert(0, "/content")
CONFIG_PATH = "/content/h3_config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("这台 VM 还没初始化：先运行 `h3 up`")

import h3_comfy  # noqa: E402

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)
launch_args = cfg.pop("launch_args")

client = h3_comfy.ComfyClient()
try:
    client.get_json("/system_stats")
except Exception:  # noqa: BLE001
    print("ComfyUI 没有响应，正在重启（之前可能因为内存不足退出，`h3 logs` 可查看）…", flush=True)
    proc = subprocess.Popen(launch_args, cwd="/content/ComfyUI", stdout=open("/content/comfyui.log", "a"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    client.wait_ready(timeout=600, proc=proc)

h3 = h3_comfy.H3Image(**cfg)
jobs = H3_JOBS  # noqa: F821  (injected by the caller)
for i, job in enumerate(jobs, 1):
    job = dict(job)
    kind = job.pop("kind")
    print(f"[{i}/{len(jobs)}] 生成中…（第一张要先把权重读进显存，会多等 1～2 分钟）" if i == 1 else f"[{i}/{len(jobs)}] 生成中…",
          flush=True)
    try:
        r = h3.generate(**job) if kind == "t2i" else h3.edit(**job)
    except (h3_comfy.ComfyError, OSError, ValueError) as e:
        print("H3_RESULT " + json.dumps({"index": i, "error": str(e)}), flush=True)
        continue
    print("H3_RESULT " + json.dumps({
        "index": i, "kind": kind, "paths": r.paths, "prompt": r.prompt, "width": r.width, "height": r.height,
        "seed": r.seed, "seconds": round(r.seconds, 1), "enhanced": r.enhanced, **r.extra,
    }), flush=True)
print("H3_DONE {}", flush=True)
