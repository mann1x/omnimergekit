#!/usr/bin/env python3
"""Does the Qwen3.6-A3B-Coder mmproj actually work on Ornith?

Method: same mmproj, same image, same prompt, two models.
  CONTROL = Qwen3.6-27B-A3B-Coder  (the model the projector was built for)
  TEST    = Ornith-1.5-27B-A3B-Coder

The control is the whole point: without it, a garbage answer from Ornith could be
blamed on CPU inference, the quant, or the harness. If the control describes the
image correctly and Ornith does not, the projector is model-specific -- proven,
not asserted.

Test image is synthetic and unambiguous: three vertical bands, red|green|blue.
A working pipeline says those three colours in that order. Nothing else does.
"""
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import zlib

W = H = 448
PNG = "/mnt/sdc/ornith_ol/vision_test_rgb.png"
MMPROJ = "/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf"
CONTROL = "/srv/ml/models/gguf/Qwen3.6-27B-A3B-Coder-MTP-GGUF/Qwen3.6-27B-A3B-Coder-IQ3_M.gguf"
TEST = "/mnt/sdc/ornith_ol/stage/Ornith-184e-Coder-Q6_K_L.gguf"
BIN = "/opt/llama.cpp/build/bin/llama-server"
PORT = 8099
PROMPT = "Look at this image. Name the colours of the vertical bands from left to right. Answer in one short sentence."


def make_png(path):
    rows = []
    for _ in range(H):
        row = bytearray([0])
        for x in range(W):
            if x < W // 3:
                row += bytes((220, 30, 30))
            elif x < 2 * W // 3:
                row += bytes((30, 190, 60))
            else:
                row += bytes((40, 60, 220))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)
    return len(png)


def wait_ready(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=5):
                return True
        except Exception:
            time.sleep(5)
    return False


def ask(port, png_b64):
    body = {
        "model": "probe",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + png_b64}},
        ]}],
        "max_tokens": 120, "temperature": 0.0,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def run(label, model):
    print("\n=== %s ===" % label, flush=True)
    print("    model  = %s" % os.path.basename(model), flush=True)
    log = open("/srv/ml/logs/vprobe_%s.log" % label, "wb")
    p = subprocess.Popen(
        [BIN, "-m", model, "--mmproj", MMPROJ, "--port", str(PORT),
         "-c", "8192", "-ngl", "0", "-t", "10", "--no-warmup"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(PORT):
            print("    SERVER DID NOT COME UP -- see /srv/ml/logs/vprobe_%s.log" % label)
            return None
        print("    server ready, sending image", flush=True)
        out = ask(PORT, base64.b64encode(open(PNG, "rb").read()).decode())
        print("    RAW ANSWER: %r" % out[:400], flush=True)
        low = out.lower()
        hits = [c for c in ("red", "green", "blue") if c in low]
        ordered = ("red" in low and "green" in low and "blue" in low
                   and low.index("red") < low.index("green") < low.index("blue"))
        print("    colours found = %s   correct order = %s" % (hits, ordered), flush=True)
        return {"answer": out, "hits": hits, "ordered": ordered}
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
        log.close()
        time.sleep(8)


def main():
    print("test image: %d bytes" % make_png(PNG))
    for f in (MMPROJ, CONTROL, TEST):
        if not os.path.exists(f):
            print("MISSING: %s" % f)
            return 2
    c = run("control_qwen", CONTROL)
    t = run("test_ornith", TEST)
    print("\n=== VERDICT ===")
    print("  control (Qwen)  :", "PASS" if c and c["ordered"] else "FAIL/none")
    print("  test    (Ornith):", "PASS" if t and t["ordered"] else "FAIL/none")
    if c and c["ordered"] and t and t["ordered"]:
        print("  => mmproj TRANSFERS to Ornith -- vision tags are viable")
    elif c and c["ordered"]:
        print("  => mmproj is MODEL-SPECIFIC: works on Qwen, not on Ornith")
    else:
        print("  => CONTROL FAILED: harness/CPU/quant issue, test is inconclusive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
