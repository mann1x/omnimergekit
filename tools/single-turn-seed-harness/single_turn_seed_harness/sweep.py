"""Single-turn seed-sweep driver.

For every (problem x seed) it POSTs one single-turn chat request to a running
OpenAI-compatible `/v1/chat/completions` endpoint and grades the raw response
with the hammer classifier (`classify.py`, copied verbatim from
`opencode_capture/hammer_raw.py`).

Sampler = FIELD condition: NO sampler params are sent (`temperature`, `top_p`,
`top_k`, `min_p`, `repeat_penalty` are all omitted), so llama-server uses its
own defaults -- exactly the `--field` condition in hammer_raw. `cache_prompt` is
sent false so every request forces a fresh prefill. Requests are non-streaming
(the classifier reads `choices[].message.{content,reasoning_content}`), which is
byte-for-byte the shape hammer_raw grades.

The template under test is chosen by WHICH server you point `--server` at -- the
harness itself is template-agnostic. Run it once per llama-server (one per
template) and diff the two summaries.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

from .classify import ALL_VERDICTS, FAIL_VERDICTS, classify

# sampler keys stripped from every request to reproduce the field default-sampler
# condition (matches hammer_raw's --field behaviour).
_SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty")


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _post_one(server, task, seed, max_tokens, timeout):
    """Fire one single-turn request; return (verdict, flags, stats, latency)."""
    body = {
        "messages": [{"role": "user", "content": task["message"]}],
        "stream": False,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "seed": seed,
    }
    for k in _SAMPLER_KEYS:
        body.pop(k, None)  # belt-and-braces: never send a sampler
    url = server.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    status = 200
    try:
        resp = urllib.request.urlopen(urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST"),
            timeout=timeout)
        raw = resp.read().decode("utf-8", "replace")
        status = resp.getcode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    except Exception as e:  # noqa: BLE001
        raw = '{"_err": %r}' % str(e)
        status = 599
    v, flags, st = classify(status, raw, max_tokens)
    return v, flags, st, round(time.time() - t0, 1)


def run_sweep(server, name, tasks, seeds, out_dir, max_tokens=16384,
              timeout=2400.0, concurrency=4, log=print):
    """Sweep every (task x seed) and write JSONL + summary.json under out_dir.

    Returns the summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "responses_%s.jsonl" % name)
    summary_path = os.path.join(out_dir, "summary_%s.json" % name)

    units = [(t, s) for t in tasks for s in seeds]
    total = len(units)
    tally = Counter()
    records = []
    lens = []          # total generated chars (content + reasoning)
    toks = []          # completion_tokens
    lock = threading.Lock()
    done = {"n": 0}

    log("=== single-turn seed sweep: %s ===" % name)
    log("server=%s  tasks=%d  seeds=%d  units=%d  max_tokens=%d  concurrency=%d"
        % (server, len(tasks), len(seeds), total, max_tokens, concurrency))
    log("sampler=field-default (no sampler sent)  cache_prompt=false  stream=false")

    jf = open(jsonl_path, "w", buffering=1)  # line-buffered so partials survive a kill

    def work(unit):
        task, seed = unit
        v, flags, st, lat = _post_one(server, task, seed, max_tokens, timeout)
        rec = {
            "name": name, "task_id": task["task_id"], "source": task["source"],
            "lang": task["lang"], "seed": seed, "verdict": v,
            "finish": st.get("fin"), "content_chars": st.get("c"),
            "reasoning_chars": st.get("r"), "completion_tokens": st.get("ct"),
            "n_tools": st.get("tools"), "latency_s": lat,
            "flags": [f[:200] for f in flags],
        }
        with lock:
            tally[v] += 1
            records.append(rec)
            if st.get("c") is not None or st.get("r") is not None:
                lens.append((st.get("c") or 0) + (st.get("r") or 0))
            if st.get("ct") is not None:
                toks.append(st["ct"])
            jf.write(json.dumps(rec) + "\n")
            done["n"] += 1
            n = done["n"]
        if n % 10 == 0 or n == total:
            bad = sum(tally[k] for k in FAIL_VERDICTS)
            log("  [%4d/%4d] %-13s %-24s %5.1fs  fin=%-8s c=%-6s r=%-6s  "
                "running fail=%d (%.1f%%)"
                % (n, total, v, task["task_id"][:24], lat, st.get("fin"),
                   st.get("c"), st.get("r"), bad, 100.0 * bad / max(1, n)))
        return rec

    try:
        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                list(ex.map(work, units))
        else:
            for u in units:
                work(u)
    finally:
        jf.close()

    lens.sort()
    toks.sort()
    bad = sum(tally[k] for k in FAIL_VERDICTS)
    summary = {
        "name": name,
        "server": server,
        "total": total,
        "n_tasks": len(tasks),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "max_tokens": max_tokens,
        "sampler": "field-default",
        "cache_prompt": False,
        "verdicts": {k: tally.get(k, 0) for k in ALL_VERDICTS if tally.get(k, 0)
                     or k in ("CLEAN",) + FAIL_VERDICTS + ("ABORT",)},
        "fail_verdicts": list(FAIL_VERDICTS),
        "fails": bad,
        "fail_rate": bad / max(1, total),
        "completion_len_chars": {
            "p50": _percentile(lens, 0.50), "p90": _percentile(lens, 0.90),
            "max": lens[-1] if lens else None, "n": len(lens),
        },
        "completion_tokens": {
            "p50": _percentile(toks, 0.50), "p90": _percentile(toks, 0.90),
            "max": toks[-1] if toks else None, "n": len(toks),
        },
        "jsonl": os.path.abspath(jsonl_path),
    }
    json.dump(summary, open(summary_path, "w"), indent=2)

    log("--> %s: %d/%d FAIL (%.1f%%)  [%s]"
        % (name, bad, total, 100.0 * summary["fail_rate"],
           "  ".join("%s=%d" % (k, tally[k]) for k in ALL_VERDICTS if tally.get(k))))
    log("    completion chars p50=%s p90=%s   tokens p50=%s p90=%s"
        % (summary["completion_len_chars"]["p50"], summary["completion_len_chars"]["p90"],
           summary["completion_tokens"]["p50"], summary["completion_tokens"]["p90"]))
    log("    wrote %s" % summary_path)
    return summary
