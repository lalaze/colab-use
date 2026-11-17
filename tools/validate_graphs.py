"""Validate h3_comfy's API graphs against a real ComfyUI, without a GPU or real weights.

    python3 tools/validate_graphs.py --comfy-dir /path/to/ComfyUI --python /path/to/venv/bin/python

Starts ComfyUI on the CPU with empty placeholder model files, then for every
graph h3_comfy builds checks that ComfyUI's validation accepts it (node
classes, input names, types and combo values). Execution is then expected to
fail at the first loader, because the placeholder weights are empty; the test
also checks that the client reports that failure as a ComfyError.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "remote"))

import h3_comfy  # noqa: E402
from h3_comfy import ComfyClient, ComfyError  # noqa: E402

PORT = 8199


def tiny_png(path: pathlib.Path, w: int = 64, h: int = 48):
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * w for _ in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    base = pathlib.Path(tempfile.mkdtemp(prefix="h3val_"))
    for _, rel in h3_comfy.MODEL_FILES.values():
        p = base / "models" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    (base / "custom_nodes").mkdir()  # ComfyUI lists it at startup when --base-directory is set
    print("log:", base / "comfyui.log")
    log = open(base / "comfyui.log", "w")
    proc = subprocess.Popen(
        [args.python, "main.py", "--cpu", "--port", str(PORT), "--disable-auto-launch",
         "--base-directory", str(base), "--output-directory", str(base / "output")],
        cwd=args.comfy_dir, stdout=log, stderr=subprocess.STDOUT)
    failures = 0
    try:
        client = ComfyClient(f"http://127.0.0.1:{PORT}")
        client.wait_ready(timeout=600, proc=proc)
        png = base / "ref.png"
        tiny_png(png)
        ref = client.upload_image(str(png))
        print("uploaded reference as", ref)

        names = {k: os.path.basename(rel) for k, (_, rel) in h3_comfy.MODEL_FILES.items()}
        t2i_template, t2i_system = h3_comfy.load_pe_prompts("t2i")
        edit_template, edit_system = h3_comfy.load_pe_prompts("edit")
        print(f"PE prompts loaded: t2i template {len(t2i_template)} chars, edit template {len(edit_template)} chars")

        graphs = {
            "t2i int8 16MP": h3_comfy.image_graph("a cat", 4096, 4096, 1, dit=names["dit_int8"],
                                                  text_encoder=names["text_encoder"], vae=names["vae"]),
            "t2i bf16 21:9": h3_comfy.image_graph("a cat", *h3_comfy.canvas_size("21:9"), 2, dit=names["dit_bf16"],
                                                  text_encoder=names["text_encoder"], vae=names["vae"]),
            "edit 2 refs": h3_comfy.image_graph("<Picture 1> and <Picture 2>", 2048, 2048, 3, dit=names["dit_int8"],
                                                text_encoder=names["text_encoder"], vae=names["vae"],
                                                reference_images=[ref, ref]),
            "enhance t2i": h3_comfy.enhance_graph("一只猫", template=t2i_template, system_prompt=t2i_system,
                                                  enhancer=names["prompt_enhancer"], seed=4),
            "enhance edit": h3_comfy.enhance_graph("把车换成冰淇淋车", template=edit_template, system_prompt=edit_system,
                                                   enhancer=names["prompt_enhancer"], seed=5, reference_images=[ref]),
        }
        for label, graph in graphs.items():
            try:
                prompt_id = client.queue(graph)
            except ComfyError as e:
                failures += 1
                print(f"FAIL  {label}: rejected by validation\n      {e}")
                continue
            try:
                client.wait(prompt_id, timeout=300, poll=0.5)
                print(f"??    {label}: accepted and ran (unexpected with empty weights)")
            except ComfyError as e:
                print(f"OK    {label}: accepted by validation; execution failed as expected -> {str(e)[:110]}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()
    print("log:", base / "comfyui.log")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
