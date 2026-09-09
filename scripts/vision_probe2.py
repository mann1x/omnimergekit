#!/usr/bin/env python3
"""Round 2: can Ornith SEE, or did it guess?

Round 1 used red|green|blue -- the most guessable possible answer. This round uses
FIVE bands in a non-obvious order (orange, purple, yellow, teal, brown) plus a
separate control question about band COUNT. A model with a dead projector cannot
guess this; a model that is actually seeing gets most of it.

Also runs a NO-IMAGE arm: same prompt, no image attached. If the no-image arm
answers with colours anyway, the 'answer' is coming from the prompt, not the pixels.
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

W = 560
H = 400
BANDS = [("orange", (235, 130, 20)), ("purple", (120, 40, 160)), ("yellow", (240, 220, 40)),
         ("teal", (20, 150, 150)), ("brown", (110, 70, 35))]
PNG = "/mnt/sdc/ornith_ol/vision_test5.png"
MMPROJ = "/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf"
TEST = "/mnt/sdc/ornith_ol/stage/Ornith-184e-Coder-Q6_K_L.gguf"
BIN = "/opt/llama.cpp/build/bin/llama-server"
PORT = 8099
Q = ("How many vertical colour bands are in this image, and what colour is each one "
     "from left to right? Answer in one short sentence.")


def make_png(path):
    n = len(BANDS)
    bw = W // n
    rows = []
    for _ in range(H):
        row = bytearray([0])
        for x in range(W):
            row += bytes(BANDS[min(x // bw, n - 1)][1])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    open(path, "wb").write(png)


def wait_ready(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=5):
                return True
        except Exception:
            time.sleep(5)
    return False


def ask(port, with_image):
    content = [{"type": "text", "text": Q}]
    if with_image:
        b64 = base64.b64encode(open(PNG, "rb").read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}})
    body = {"model": "probe", "messages": [{"role": "user", "content": content}],
            "max_tokens": 150, "temperature": 0.0}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or "")


def score(ans):
    low = ans.lower()
    hits = [c for c, _ in BANDS if c in low]
    five = ("five" in low or "5" in low)
    return hits, five


def main():
    make_png(PNG)
    print("image: 5 bands = %s" % ", ".join(c for c, _ in BANDS), flush=True)
    log = open("/srv/ml/logs/vprobe2.log", "wb")
    p = subprocess.Popen([BIN, "-m", TEST, "--mmproj", MMPROJ, "--port", str(PORT),
                          "-c", "8192", "-ngl", "0", "-t", "10", "--no-warmup"],
                         stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(PORT):
            print("server did not start"); return 2
        for label, img in (("WITH image", True), ("NO image (control)", False)):
            a = ask(PORT, img)
            h, five = score(a)
            print("\n--- %s" % label, flush=True)
            print("    answer : %r" % a[:300], flush=True)
            print("    matched: %s   said-five: %s   (%d/5)" % (h, five, len(h)), flush=True)
    finally:
        p.terminate()
        try: p.wait(timeout=60)
        except Exception: p.kill()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
