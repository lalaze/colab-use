# MiniMax-H3-Image bootstrap. `h3 up` runs this inside the Colab VM's kernel
# (via `colab exec`), prepending two variables:
#   H3_CFG       dict: dit, enhancer, comfy_version, check_location
#   H3_COMFY_SRC str:  source of h3_comfy.py
# It is idempotent: re-running skips what is already installed/downloaded/running.

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

COMFY_DIR = "/content/ComfyUI"
MODELS_DIR = f"{COMFY_DIR}/models"
OUTPUT_DIR = "/content/outputs"
CONFIG_PATH = "/content/h3_config.json"
LOG_PATH = "/content/comfyui.log"
GIB = 1024 ** 3
EXCLUDED = {"US", "GB", "KR", "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}

t_start = time.time()


def step(msg):
    print(f"\n== {msg}", flush=True)


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- client module
with open("/content/h3_comfy.py", "w", encoding="utf-8") as f:
    f.write(H3_COMFY_SRC)  # noqa: F821  (injected by the caller)
if "/content" not in sys.path:
    sys.path.insert(0, "/content")
import importlib  # noqa: E402

import h3_comfy  # noqa: E402

importlib.reload(h3_comfy)
cfg = H3_CFG  # noqa: F821  (injected by the caller)

# ---------------------------------------------------------------- hardware
step("检查硬件")
smi = shutil.which("nvidia-smi") and subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
    capture_output=True, text=True)
if not smi or smi.returncode != 0:
    raise RuntimeError("这台 VM 没有 GPU。用 `h3 down` 关掉，再 `h3 up --gpu A100`。")
gpu_name, vram_mib, driver = [s.strip() for s in smi.stdout.strip().splitlines()[0].split(",")]
vram_gib = int(vram_mib) / 1024
ram_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
disk_free_gb = shutil.disk_usage("/content").free / 1e9
print(f"GPU  {gpu_name}  {vram_gib:.1f} GiB (driver {driver})")
print(f"RAM  {ram_gib:.1f} GiB   磁盘可用 {disk_free_gb:.0f} GB")
if vram_gib < 22 or "T4" in gpu_name:
    raise RuntimeError("这张卡跑不动 MiniMax-H3-Image（T4 不支持 bf16，显存也不够）。`h3 down` 后换 A100 / G4 / H100。")
if vram_gib < 39:
    print("⚠️  显存 < 40 GB：DiT 放不进显存，会大量 offload，很慢也很费 CU。建议换 A100 80GB / G4 / H100。")

dit = cfg["dit"]
if dit == "bf16" and vram_gib < 70:
    print("⚠️  bf16 DiT 需要 ≥70 GB 显存，自动改用 int8。")
    dit = "int8"
keys = [f"dit_{dit}", "text_encoder", "vae"] + (["prompt_enhancer"] if cfg["enhancer"] else [])
need_gb = sum(h3_comfy.MODEL_SIZES_GB[k] for k in keys)
print(f"计划  DiT={dit}  提示词扩写={'开' if cfg['enhancer'] else '关'}  权重 {need_gb:.0f} GB")
if vram_gib < need_gb * 1e9 / GIB + 24:
    print("     显存放不下全部权重 + 16MP 的中间结果，ComfyUI 会自动在显存/内存之间换入换出（会慢一些）。")
if disk_free_gb < need_gb + 15:
    raise RuntimeError(f"磁盘不够：需要约 {need_gb + 15:.0f} GB，可用 {disk_free_gb:.0f} GB。")

if cfg["check_location"]:
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=10) as r:
            loc = json.load(r)
        print(f"VM 位置  {loc.get('country')} / {loc.get('region')} / {loc.get('org')}")
        if loc.get("country") in EXCLUDED:
            print("⚠️  这台 VM 在 MiniMax-H3-Image 许可证的排除地域（美/欧/英/韩）内，请先确认许可证是否允许你的使用方式。")
    except Exception as e:  # noqa: BLE001
        print("VM 位置查询失败:", e)

# ---------------------------------------------------------------- ComfyUI
step(f"安装 ComfyUI {cfg['comfy_version']}")
if not os.path.isdir(COMFY_DIR):
    sh(["git", "clone", "-q", "--depth", "1", "--branch", cfg["comfy_version"],
        "https://github.com/comfyanonymous/ComfyUI.git", COMFY_DIR])
else:
    print("已存在，跳过 clone")
sh([sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY_DIR}/requirements.txt", "huggingface_hub[hf_xet]"])
subprocess.run([sys.executable, "-c",
                "import comfyui_version as v, torch; print('ComfyUI', v.__version__, '| torch', torch.__version__,"
                " '| CUDA', torch.version.cuda, '| bf16', torch.cuda.is_bf16_supported())"], cwd=COMFY_DIR)

# ---------------------------------------------------------------- weights
step(f"下载模型权重（{need_gb:.0f} GB，直接下到 VM 本地 SSD）")


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _progress(stop, t0, base):
    while not stop.wait(20):
        done = (_dir_bytes(MODELS_DIR) - base) / 1e9
        print(f"  … 已下载 {done:.1f} GB（{done / max(time.time() - t0, 1):.2f} GB/s）", flush=True)


t0, base, stop = time.time(), _dir_bytes(MODELS_DIR), threading.Event()
threading.Thread(target=_progress, args=(stop, t0, base), daemon=True).start()
try:
    # Runs in a fresh interpreter so hf_xet (pip-installed above) is actually used; see h3_comfy.download_models.
    paths = h3_comfy.download_models(keys, MODELS_DIR)
finally:
    stop.set()
for k in keys:
    print(f"  {os.path.getsize(paths[k]) / 1e9:6.2f} GB  {os.path.basename(paths[k])}")
print(f"  完成，用时 {(time.time() - t0) / 60:.1f} 分钟（已存在的文件会跳过）")

# ---------------------------------------------------------------- server
step("启动 ComfyUI")
# ComfyUI's default dynamic-VRAM management keeps weights on the GPU while they fit.
launch_args = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
               "--output-directory", OUTPUT_DIR, "--disable-auto-launch", *cfg.get("extra_args", [])]
client = h3_comfy.ComfyClient()
proc = None
try:
    client.get_json("/system_stats")
    print("ComfyUI 已在运行（要换参数：h3 restart）")
except Exception:  # noqa: BLE001
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    proc = subprocess.Popen(launch_args, cwd=COMFY_DIR, stdout=open(LOG_PATH, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
stats = client.wait_ready(timeout=600, proc=proc)
dev = stats["devices"][0]
print(f"ComfyUI {stats['system'].get('comfyui_version')} | {dev['name']} | 显存空闲 {dev['vram_free'] / GIB:.1f} GiB")
warn = subprocess.run(["grep", "-iE", "warn|error|kitchen|aimdo", LOG_PATH], capture_output=True, text=True).stdout
if warn.strip():
    print("日志里值得留意的行：\n" + "\n".join(warn.strip().splitlines()[-12:]))

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "dit_file": paths[f"dit_{dit}"],
        "text_encoder_file": paths["text_encoder"],
        "vae_file": paths["vae"],
        "enhancer_file": paths.get("prompt_enhancer"),
        "comfy_output_dir": OUTPUT_DIR,
        "launch_args": launch_args,
    }, f, indent=1)

print("H3_DONE " + json.dumps({"gpu": gpu_name, "vram_gib": round(vram_gib, 1), "dit": dit,
                               "enhancer": bool(cfg["enhancer"]), "minutes": round((time.time() - t_start) / 60, 1)}),
      flush=True)
