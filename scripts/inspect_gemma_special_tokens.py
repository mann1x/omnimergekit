#!/usr/bin/env python3
"""inspect_gemma_special_tokens.py <gguf> [<gguf> ...]

Dump the token_type of Gemma-4's novel angle-pipe delimiter tokens
(<|channel>, <channel|>, <|turn>, <turn|>, <|think|>, <|tool*>, ...) to test whether
they are mistyped as NORMAL (=> the model can emit them as ordinary text and loop on
them, with no EOG/control handling) vs CONTROL/USER_DEFINED. Compare a looping model
(12B) against a passing one (26B-A4B) to see if the vocab typing differs.
"""
import sys
import numpy as np
from gguf import GGUFReader

TYPE = {0: "UNSPEC", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL",
        4: "USER_DEF", 5: "UNUSED", 6: "BYTE"}
NEED = ["channel", "turn", "think", "thought", "tool", "image", "audio",
        "video", "start_of", "end_of"]


def read_field_array(f):
    """Robustly return a python list for a gguf array field (strings or ints)."""
    # newer gguf: ReaderField.contents() returns the full python value
    try:
        c = f.contents()
        if isinstance(c, (list, tuple)) and len(c) > 1:
            return list(c)
    except Exception:
        pass
    # string array: one part per element, indexed by f.data
    try:
        return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
    except Exception:
        pass
    # int array: values may be one-per-part (indexed by f.data) or one flat part
    try:
        return [int(f.parts[i][0]) for i in f.data]
    except Exception:
        return [int(x) for x in f.parts[-1]]


def scalar(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    try:
        return int(f.parts[-1][0])
    except Exception:
        try:
            return f.contents()
        except Exception:
            return "?"


def inspect(path):
    print("\n##### %s" % path)
    r = GGUFReader(path)
    toks = read_field_array(r.fields["tokenizer.ggml.tokens"])
    types = read_field_array(r.fields["tokenizer.ggml.token_type"])
    print("ntok=%d  ntype=%d  %s" % (
        len(toks), len(types),
        "LEN-MISMATCH!" if len(toks) != len(types) else "ok"))
    for k in ["bos_token_id", "eos_token_id", "eot_token_id",
              "unknown_token_id", "padding_token_id"]:
        v = scalar(r, "tokenizer.ggml." + k)
        if v is not None:
            who = toks[v] if isinstance(v, int) and 0 <= v < len(toks) else "?"
            print("  %s = %s  (%r)" % (k, v, who))
    print("  idx\ttype\ttoken")
    n = min(len(toks), len(types))
    for i in range(n):
        s = toks[i]
        special = (any(k in s for k in NEED)
                   or (("<|" in s or "|>" in s) and len(s) <= 22)
                   or s in ("</s>", "<eos>", "<bos>", "<pad>", "<unk>"))
        if special:
            t = types[i]
            tt = TYPE.get(int(t), t) if isinstance(t, (int, np.integer)) else t
            print("  %d\t%s\t%r" % (i, tt, s))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            inspect(p)
        except Exception as e:
            print("ERROR on %s: %r" % (p, e))
