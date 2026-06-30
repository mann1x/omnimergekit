#!/usr/bin/env python3
"""analyze.py — tabulate loop rates and AUDIT loop verdicts for faithfulness.

- tabulate: per-label loop-rate table from <out>/sessions/INDEX.jsonl
            (LOOP set = {DEGENERATE, TOOL_LOOP}; everything else is non-loop).
- audit:    re-derive every TOOL_LOOP/DEGENERATE verdict from the UNTRUNCATED wire log,
            catching the two artifact risks — a TOOL_LOOP that only exists because args
            were truncated to a shared 200-char prefix, and a DEGENERATE that is actually
            a coherent long answer (RUNAWAY) or a quoted token (CORRUPT).
"""
import collections
import glob
import json
import os
import re
import zlib

from .compact import classify_resp, RUNAWAY_MIN_COMPLETION, THINK_CAP, CORRUPT, _usage

LOOP = {"DEGENERATE", "TOOL_LOOP"}


def _index_rows(out_dir):
    idx = os.path.join(out_dir, "sessions", "INDEX.jsonl")
    if not os.path.isfile(idx):
        return []
    return [json.loads(l) for l in open(idx) if l.strip()]


def tabulate(out_dir):
    rows = _index_rows(out_dir)
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r.get("model_label", "?"), []).append(r)
    print("%-34s %3s %7s   verdict mix" % ("ARM (model_label)", "n", "LOOP%"))
    for lbl, ds in by.items():
        mix = collections.Counter(d.get("verdict", "?") for d in ds)
        nloop = sum(mix[v] for v in LOOP)
        n = len(ds)
        pct = (100.0 * nloop / n) if n else 0.0
        mixs = ", ".join("%s:%d" % (k, v) for k, v in sorted(mix.items(), key=lambda x: -x[1]))
        print("%-34s %3d %6.1f%%   %s" % (lbl[:34], n, pct, mixs))


def _raw_resps(sdir):
    out = {}
    for fp in sorted(glob.glob(os.path.join(sdir, "wirelog", "session-*.jsonl"))):
        for line in open(fp, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("dir") == "response" and o.get("rid") is not None:
                out[o["rid"]] = o
    return out


def _loopiness(text):
    if not text:
        return None, None
    b = text.encode("utf-8", "replace")
    zr = len(zlib.compress(b, 6)) / max(1, len(b))      # low = highly repetitive
    lines = [l for l in text.splitlines() if l.strip()]
    rl = (1 - len(set(lines)) / len(lines)) if lines else 0.0
    return zr, rl


def _full_tool_loop(resps):
    sigs = []
    for rid in sorted(resps):
        tcs = resps[rid].get("tool_calls") or []
        sigs.append((rid, (tcs[0].get("name"), tcs[0].get("arguments") or "") if tcs else None))
    best, cur, prev, span, bspan, bsig = 0, 0, object(), [], [], None
    for rid, s in sigs:
        if s is not None and s == prev:
            cur += 1; span.append(rid)
        else:
            cur = 1 if s is not None else 0
            span = [rid] if s is not None else []
            prev = s
        if cur > best:
            best, bspan, bsig = cur, list(span), s
    return best, bspan, bsig


def _trunc_tool_loop(resps):
    sigs = []
    for rid in sorted(resps):
        tcs = resps[rid].get("tool_calls") or []
        sigs.append((rid, (tcs[0].get("name"), (tcs[0].get("arguments") or "")[:200]) if tcs else None))
    best, cur, prev = 0, 0, object()
    for rid, s in sigs:
        cur = cur + 1 if (s is not None and s == prev) else (1 if s is not None else 0)
        prev = s
        best = max(best, cur)
    return best


def audit(out_dir, label=None, session_id=None):
    rows = _index_rows(out_dir)
    sel = [r for r in rows if r.get("verdict") in LOOP
           and (label is None or r.get("model_label") == label)
           and (session_id is None or r.get("session_id") == session_id)]
    sel.sort(key=lambda r: (r.get("model_label", ""), r.get("session_id", "")))
    print("Auditing %d LOOP-verdict session(s)\n" % len(sel))
    suspects = []
    for r in sel:
        lbl, sid, v = r["model_label"], r["session_id"], r["verdict"]
        resps = _raw_resps(os.path.join(out_dir, "sessions", sid))
        print("=" * 100)
        print("%-26s %s  verdict=%s" % (lbl, sid, v))
        if v == "TOOL_LOOP":
            fb, fspan, fsig = _full_tool_loop(resps)
            tb = _trunc_tool_loop(resps)
            name, args = (fsig or (None, ""))
            faith = fb >= 4
            tag = "FAITHFUL" if faith else "SUSPECT (full-arg run<4 -> 200-char collision)"
            print("  full-arg max-run=%d  trunc-arg max-run=%d  -> %s" % (fb, tb, tag))
            print("  repeated tool: %s   rids=%s" % (name, fspan))
            print("  args(full,%dch): %s" % (len(args), args[:240] + ("..." if len(args) > 240 else "")))
            if not faith:
                suspects.append((lbl, sid, v, "tool-loop vanishes on full args (run=%d)" % fb))
        else:  # DEGENERATE
            degen = [(rid, classify_resp(resps[rid]), resps[rid]) for rid in sorted(resps)
                     if classify_resp(resps[rid]) in ("RUNAWAY", "THINK_EXPLODE", "CORRUPT")]
            print("  degenerate turns: %d" % len(degen))
            for rid, vv, rp in degen:
                cont = rp.get("content") or ""
                reas = rp.get("reasoning_content") or ""
                comp, _ = _usage(rp)
                fin = rp.get("finish_reason")
                blob = reas if len(reas) >= len(cont) else cont
                zr, rl = _loopiness(blob)
                faith = True
                if vv == "CORRUPT":
                    m = next((re.search(p, b) for p in CORRUPT for b in (cont, reas) if re.search(p, b)), None)
                    pat = next((p for p in CORRUPT for b in (cont, reas) if re.search(p, b)), "?")
                    ctx = ""
                    if m:
                        b = m.string; i = m.start()
                        ctx = b[max(0, i - 60):i + 60].replace("\n", "/")
                    note = "pattern=%r ctx=...%s..." % (pat, ctx)
                    if ctx and (("`" in ctx) or ('"%s' % pat in ctx)):
                        faith = False
                elif vv == "RUNAWAY":
                    note = "fin=%s comp_tok=%s len(c)=%d len(r)=%d zlib=%.3f rep_line=%.2f" % (
                        fin, comp, len(cont), len(reas), zr or 0, rl or 0)
                    if (zr or 1) > 0.32 and (rl or 0) < 0.25:
                        faith = False
                else:  # THINK_EXPLODE
                    note = "len(c)=%d len(r)=%d (>%d) zlib=%.3f rep_line=%.2f fin=%s" % (
                        len(cont), len(reas), THINK_CAP, zr or 0, rl or 0, fin)
                print("   rid %-4s %-13s %s  [%s]" % (rid, vv, note, "FAITHFUL" if faith else "SUSPECT"))
                print("      head: %s" % blob[:200].replace("\n", "/"))
                print("      tail: %s" % blob[-200:].replace("\n", "/"))
                if not faith:
                    suspects.append((lbl, sid, v, "%s rid%s %s" % (vv, rid, note)))
    print("\n" + "=" * 100)
    if suspects:
        print("SUSPECT detections (%d) -- need eyeball:" % len(suspects))
        for s in suspects:
            print("  %-26s %s  %s :: %s" % s)
    else:
        print("ALL %d LOOP verdict(s) FAITHFUL." % len(sel))
    return suspects
