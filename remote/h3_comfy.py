"""MiniMax-H3-Image client for a local ComfyUI server.

Builds ComfyUI API-format graphs that mirror the official Comfy-Org templates
(`image_minimax_h3_image_t2i.json` / `image_minimax_h3_image_edit.json`),
queues them over HTTP and collects the saved PNGs. Standard library only;
Pillow is used when available (Colab has it) to read reference image sizes.
`download_models` runs huggingface_hub in a child process (`python h3_comfy.py download ...`).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field

DEFAULT_URL = "http://127.0.0.1:8188"

# (repo_id, path inside repo == path under ComfyUI/models)
MODEL_FILES = {
    "dit_int8": ("Comfy-Org/MiniMax-H3-Image", "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"),
    "dit_bf16": ("Comfy-Org/MiniMax-H3-Image", "diffusion_models/minimax_h3_image_pruned_bf16.safetensors"),
    "text_encoder": ("Comfy-Org/MiniMax-H3-Image", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "vae": ("Comfy-Org/MiniMax-H3-Image", "vae/minimax_h3_image_vae_fp16.safetensors"),
    "prompt_enhancer": ("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"),
}

# Approximate download sizes in GB (decimal), for disk checks and planning.
MODEL_SIZES_GB = {
    "dit_int8": 20.98,
    "dit_bf16": 40.23,
    "text_encoder": 27.14,
    "vae": 1.17,
    "prompt_enhancer": 7.34,
}

# Recommended ~16 MP canvases from the MiniMax-H3-Image model card.
SIZES_16MP = {
    "1:1": (4096, 4096),
    "4:3": (4736, 3552),
    "3:4": (3552, 4736),
    "3:2": (5024, 3360),
    "2:3": (3360, 5024),
    "16:9": (5472, 3072),
    "9:16": (3072, 5472),
    "21:9": (6272, 2688),
}

CANVAS_MULTIPLE = 32
MP = 1024 * 1024  # the model card's "16 MP" is 4096x4096

TEMPLATE_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}.json"
PE_TEMPLATES = {"t2i": "image_minimax_h3_image_t2i", "edit": "image_minimax_h3_image_edit"}


# --------------------------------------------------------------------------- weights

def download_models(keys: list[str], models_dir: str, log=lambda s: print(s, flush=True)) -> dict[str, str]:
    """Download MODEL_FILES[keys] into models_dir (parallel) in a fresh Python process; returns key -> path.

    Why a child process: in the Colab kernel huggingface_hub may already have cached
    "hf_xet not installed" before setup pip-installs it, which silently forces slow
    plain-HTTP downloads; and in IPython it looks up an HF_TOKEN Colab secret, which
    times out under `colab exec`. A plain interpreter avoids both. HF_TOKEN from the
    environment is still honoured.
    """
    import subprocess
    import sys

    spec = {k: MODEL_FILES[k] for k in keys}
    env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "download", json.dumps(spec), models_dir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    paths = None
    for line in proc.stdout:
        if line.startswith("H3_PATHS "):
            paths = json.loads(line[len("H3_PATHS "):])
        elif line.strip():
            log(line.rstrip("\n"))
    if proc.wait() != 0 or paths is None:
        raise RuntimeError(f"weight download failed (exit {proc.returncode}); see the output above")
    return paths


def _download_cli(spec_json: str, models_dir: str):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    try:
        from huggingface_hub.utils._runtime import is_xet_available

        print("  hf_xet：" + ("已启用" if is_xet_available() else "未启用（退回普通 HTTP，会慢很多）"), flush=True)
    except ImportError:
        pass
    disable_progress_bars()
    spec = json.loads(spec_json)

    def fetch(item):
        key, (repo, path) = item
        return key, hf_hub_download(repo_id=repo, filename=path, local_dir=models_dir)

    with ThreadPoolExecutor(max_workers=len(spec)) as pool:
        paths = dict(pool.map(fetch, spec.items()))
    print("H3_PATHS " + json.dumps(paths), flush=True)


# --------------------------------------------------------------------------- sizes

def parse_aspect(aspect: str) -> tuple[float, float]:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", aspect or "")
    if not m:
        raise ValueError(f"aspect ratio must look like '16:9', got {aspect!r}")
    w, h = float(m.group(1)), float(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid aspect ratio {aspect!r}")
    return w, h


def _round32(x: float) -> int:
    return max(CANVAS_MULTIPLE, int(round(x / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def canvas_size(aspect: str = "1:1", megapixels: float = 16) -> tuple[int, int]:
    """Width/height for an aspect ratio at a target area, sides multiples of 32.

    At 16 MP the official table is used. The model works best at ~16 MP and
    is effective from ~4 MP to ~20 MP.
    """
    key = aspect.replace(" ", "")
    if megapixels == 16 and key in SIZES_16MP:
        return SIZES_16MP[key]
    rw, rh = parse_aspect(aspect)
    area = megapixels * MP
    w = math.sqrt(area * rw / rh)
    return _round32(w), _round32(area / w)


def size_like(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Canvas with the aspect ratio of (width, height) at the target area."""
    return canvas_size(f"{width}:{height}", megapixels)


def nearest_aspect(aspect: str) -> str:
    """Map a free-form ratio (e.g. from the prompt enhancer) to the official table."""
    try:
        rw, rh = parse_aspect(aspect)
    except ValueError:
        return "1:1"
    target = math.log(rw / rh)
    return min(SIZES_16MP, key=lambda k: abs(math.log(SIZES_16MP[k][0] / SIZES_16MP[k][1]) - target))


# --------------------------------------------------------------------------- graphs

def _loaders(dit: str, text_encoder: str, vae: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }


def image_graph(prompt: str, width: int, height: int, seed: int, *, dit: str, text_encoder: str, vae: str,
                filename_prefix: str = "h3", reference_images: list[str] | None = None) -> dict:
    """Text-to-image, or an edit when `reference_images` (ComfyUI input filenames) is given.

    Same sampling as the official template: 6 steps, euler/simple, cfg 1,
    negative = zeroed positive (the checkpoint is guidance-distilled).
    """
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"width/height must be multiples of {CANVAS_MULTIPLE}, got {width}x{height}")
    g = _loaders(dit, text_encoder, vae)
    cond_inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width, "height": height}
    for i, name in enumerate(reference_images or [], start=1):
        node = str(100 + i)
        g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond_inputs[f"images.image_{i}"] = [node, 0]
    g["5"] = {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond_inputs}
    g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["5", 1],
        "seed": int(seed), "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
    }}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}}
    return g


def enhance_graph(user_text: str, *, template: str, system_prompt: str, enhancer: str, seed: int,
                  reference_images: list[str] | None = None) -> dict:
    """Prompt-expansion graph (Generate Text node), same settings as the official template."""
    inputs = {
        "clip": ["1", 0],
        "prompt": template.replace("{input text}", user_text),
        "system_prompt": system_prompt,
        "max_length": 8192,
        "sampling_mode": "on",
        "sampling_mode.temperature": 1.0,
        "sampling_mode.top_k": 20,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.0,
        "sampling_mode.repetition_penalty": 1.0,
        "sampling_mode.presence_penalty": 0.0,
        "sampling_mode.seed": int(seed),
        "thinking": True,
        "use_default_template": True,
        "mtp": "auto",
    }
    g = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": enhancer, "type": "minimax", "device": "default"}}}
    for i, name in enumerate(reference_images or [], start=1):
        node = str(100 + i)
        g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"images.image{i}"] = [node, 0]
    g["2"] = {"class_type": "TextGenerateV2", "inputs": inputs}
    g["3"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}
    return g


# --------------------------------------------------------------------------- PE prompts

def _walk_nodes(workflow: dict):
    yield from workflow.get("nodes", [])
    for sg in workflow.get("definitions", {}).get("subgraphs", []):
        yield from sg.get("nodes", [])


def extract_pe_prompts(workflow: dict) -> tuple[str, str]:
    """(template containing '{input text}', system prompt) from an official template workflow."""
    template = system = None
    for node in _walk_nodes(workflow):
        if node.get("type") != "PrimitiveStringMultiline":
            continue
        value = (node.get("widgets_values") or [""])[0]
        if not isinstance(value, str):
            continue
        if "{input text}" in value:
            template = value
        elif value.startswith("Reason in the thinking section"):
            system = value
    if not template or not system:
        raise RuntimeError("prompt-enhancer prompts not found in the workflow template")
    return template, system


def load_pe_prompts(kind: str) -> tuple[str, str]:
    """Load the official prompt-enhancer prompts ('t2i' or 'edit').

    Prefers the copy bundled with ComfyUI's pip-installed template packages,
    falls back to the Comfy-Org/workflow_templates repo on GitHub.
    """
    name = PE_TEMPLATES[kind]
    import pathlib
    import sys

    for entry in sys.path:
        base = pathlib.Path(entry or ".")
        if not base.is_dir():
            continue
        for pkg in base.glob("comfyui_workflow_templates*"):
            for hit in pkg.rglob(f"{name}.json"):
                try:
                    return extract_pe_prompts(json.loads(hit.read_text(encoding="utf-8")))
                except (OSError, RuntimeError, ValueError):
                    continue
    with urllib.request.urlopen(TEMPLATE_URL.format(name=name), timeout=60) as r:
        return extract_pe_prompts(json.loads(r.read().decode("utf-8")))


def parse_pe_output(text: str) -> tuple[str, str | None]:
    """(expanded_caption, aspect_ratio or None) from the enhancer's final answer."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("expanded_caption"), str):
            ar = data.get("aspect_ratio")
            return data["expanded_caption"].strip(), ar if isinstance(ar, str) else None
    raise ValueError(f"prompt enhancer returned no JSON caption: {text[:300]!r}")


# --------------------------------------------------------------------------- HTTP client

class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str = DEFAULT_URL):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 60):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise ComfyError(f"HTTP {e.code} on {path}: {body[:4000]}") from None

    def get_json(self, path: str):
        return json.loads(self._request(path))

    def wait_ready(self, timeout: float = 300, proc=None) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                raise ComfyError(f"ComfyUI exited with code {proc.returncode}; see the log")
            try:
                return self.get_json("/system_stats")
            except (OSError, ComfyError):
                time.sleep(2)
        raise ComfyError(f"ComfyUI did not answer on {self.url} within {timeout}s")

    def queue(self, graph: dict) -> str:
        body = json.dumps({"prompt": graph, "client_id": self.client_id}).encode("utf-8")
        resp = json.loads(self._request("/prompt", body, {"Content-Type": "application/json"}))
        if resp.get("node_errors"):
            raise ComfyError(f"graph rejected: {json.dumps(resp['node_errors'], ensure_ascii=False)[:4000]}")
        return resp["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 3600, poll: float = 1.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self.get_json(f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    detail = msgs[-1][1] if msgs else status
                    if isinstance(detail, dict):
                        detail = f"{detail.get('node_type')}: {detail.get('exception_message')}"
                    raise ComfyError(f"execution failed: {detail}")
                return entry
            time.sleep(poll)
        raise ComfyError(f"timed out after {timeout}s waiting for {prompt_id}")

    def run(self, graph: dict, timeout: float = 3600) -> dict:
        return self.wait(self.queue(graph), timeout=timeout)

    def upload_image(self, path: str) -> str:
        """Upload a local image to ComfyUI's input folder; returns the name LoadImage expects."""
        boundary = uuid.uuid4().hex
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            payload = f.read()
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8"),
            payload,
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        resp = json.loads(self._request("/upload/image", b"".join(parts),
                                        {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return f"{resp['subfolder']}/{resp['name']}" if resp.get("subfolder") else resp["name"]

    def interrupt(self):
        self._request("/interrupt", b"", {"Content-Type": "application/json"})

    def free_memory(self, unload_models: bool = True):
        body = json.dumps({"unload_models": unload_models, "free_memory": True}).encode("utf-8")
        self._request("/free", body, {"Content-Type": "application/json"})


def history_images(entry: dict) -> list[dict]:
    return [img for out in entry.get("outputs", {}).values() for img in out.get("images", [])
            if img.get("type") == "output"]


def history_text(entry: dict) -> str:
    for out in entry.get("outputs", {}).values():
        if out.get("text"):
            return "".join(out["text"])
    return ""


# --------------------------------------------------------------------------- high level

@dataclass
class Result:
    paths: list[str]
    prompt: str
    width: int
    height: int
    seed: int
    seconds: float
    enhanced: bool = False
    extra: dict = field(default_factory=dict)


class H3Image:
    """High-level MiniMax-H3-Image runner on top of a running ComfyUI."""

    def __init__(self, *, dit_file: str, text_encoder_file: str, vae_file: str, comfy_output_dir: str,
                 enhancer_file: str | None = None, save_dir: str | None = None, url: str = DEFAULT_URL):
        self.client = ComfyClient(url)
        self.dit = os.path.basename(dit_file)
        self.text_encoder = os.path.basename(text_encoder_file)
        self.vae = os.path.basename(vae_file)
        self.enhancer = os.path.basename(enhancer_file) if enhancer_file else None
        self.comfy_output_dir = comfy_output_dir
        self.save_dir = save_dir
        self._pe_prompts: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _seed(seed: int | None) -> int:
        return random.randrange(2**32) if seed is None or seed < 0 else int(seed)

    def _collect(self, entry: dict, tag: str, seed: int) -> list[str]:
        """Copy outputs to save_dir under unique names.

        ComfyUI's own counter (h3_t2i_00001_.png) restarts on every new VM, so
        reusing it on Drive would overwrite images from earlier sessions.
        """
        paths = []
        for img in history_images(entry):
            src = os.path.join(self.comfy_output_dir, img.get("subfolder", ""), img["filename"])
            if not self.save_dir:
                paths.append(src)
                continue
            os.makedirs(self.save_dir, exist_ok=True)
            stem = f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}_s{seed}"
            dst, n = os.path.join(self.save_dir, f"{stem}.png"), 1
            while os.path.exists(dst):
                n += 1
                dst = os.path.join(self.save_dir, f"{stem}_{n}.png")
            shutil.copy2(src, dst)
            paths.append(dst)
        if not paths:
            raise ComfyError("the graph finished without saving an image")
        return paths

    def enhance(self, text: str, *, reference_images: list[str] | None = None, seed: int | None = None,
                _uploaded: list[str] | None = None) -> tuple[str, str | None]:
        """Rewrite a short request into a detailed English caption (+ suggested aspect ratio)."""
        if not self.enhancer:
            raise ComfyError("prompt enhancer weights were not downloaded (ENABLE_PROMPT_ENHANCER=False)")
        kind = "edit" if (reference_images or _uploaded) else "t2i"
        if kind not in self._pe_prompts:
            self._pe_prompts[kind] = load_pe_prompts(kind)
        template, system = self._pe_prompts[kind]
        uploaded = _uploaded if _uploaded is not None else [self.client.upload_image(p) for p in reference_images or []]
        entry = self.client.run(enhance_graph(text, template=template, system_prompt=system,
                                              enhancer=self.enhancer, seed=self._seed(seed),
                                              reference_images=uploaded))
        return parse_pe_output(history_text(entry))

    def generate(self, prompt: str, *, aspect: str = "1:1", megapixels: float = 16, width: int | None = None,
                 height: int | None = None, seed: int | None = None, enhance: bool = False,
                 filename_prefix: str = "h3_t2i") -> Result:
        t0 = time.time()
        final_prompt, extra = prompt, {}
        if enhance:
            final_prompt, suggested = self.enhance(prompt, seed=seed)
            extra["suggested_aspect"] = suggested
            if suggested and not (width and height) and aspect == "auto":
                aspect = nearest_aspect(suggested)
        if aspect == "auto":
            aspect = "1:1"
        if not (width and height):
            width, height = canvas_size(aspect, megapixels)
        seed = self._seed(seed)
        entry = self.client.run(image_graph(final_prompt, width, height, seed, dit=self.dit,
                                            text_encoder=self.text_encoder, vae=self.vae,
                                            filename_prefix=filename_prefix))
        return Result(self._collect(entry, filename_prefix, seed), final_prompt, width, height, seed,
                      time.time() - t0, enhance, extra)

    def edit(self, prompt: str, images: list[str], *, megapixels: float = 16, width: int | None = None,
             height: int | None = None, seed: int | None = None, enhance: bool = False,
             filename_prefix: str = "h3_edit") -> Result:
        """Edit/compose from reference images; refer to them as <Picture 1>, <Picture 2>, ... (max 16)."""
        if not images:
            raise ValueError("edit needs at least one reference image")
        if len(images) > 16:
            raise ValueError("MiniMax-H3-Image supports up to 16 reference images")
        t0 = time.time()
        uploaded = [self.client.upload_image(p) for p in images]
        final_prompt, extra = prompt, {}
        if enhance:
            final_prompt, extra["suggested_aspect"] = self.enhance(prompt, seed=seed, _uploaded=uploaded)
        if not (width and height):
            try:
                from PIL import Image

                with Image.open(images[0]) as im:
                    width, height = size_like(im.width, im.height, megapixels)
            except ImportError:
                width, height = canvas_size("1:1", megapixels)
        seed = self._seed(seed)
        entry = self.client.run(image_graph(final_prompt, width, height, seed, dit=self.dit,
                                            text_encoder=self.text_encoder, vae=self.vae,
                                            filename_prefix=filename_prefix, reference_images=uploaded))
        return Result(self._collect(entry, filename_prefix, seed), final_prompt, width, height, seed,
                      time.time() - t0, enhance, extra)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "download":
        _download_cli(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: python h3_comfy.py download '<json {key: [repo, path]}>' <models_dir>")
