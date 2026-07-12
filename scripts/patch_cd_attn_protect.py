#!/usr/bin/env python3
"""Patch generate_cd_maps.py: extend attention protection from attn_v/attn_k to
the FULL attention (add attn_q + attn_output). Root cause of v6/v7 CD-Q4_K_M
rumination: attn_q.weight + attn_output.weight sat in BLOCK_TENSOR_ROLES, so on
the 22 'low'-tier layers they dropped to IQ3_S (3.4-bit) → stop-token collapse.
Fix: protect all 4 attention projections at the per-tier ATTN_VK_PROTECT value.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                 else "/srv/ml/repos/omnimergekit/scripts/generate_cd_maps.py")
src = p.read_text()
orig = src

# 1. Remove attn_q.weight + attn_output.weight from the per-layer role list.
roles_block = '    "attn_q.weight",\n    "attn_output.weight",\n'
if roles_block not in src:
    sys.exit("FATAL: BLOCK_TENSOR_ROLES attn_q/attn_output lines not found (already patched?)")
src = src.replace(roles_block, "")

# 2. Emit attn_q + attn_output explicitly at the protect value, every layer.
protect_old = ('        lines.append(f"blk.{li}.attn_v.weight={attn_vk}")\n'
               '        lines.append(f"blk.{li}.attn_k.weight={attn_vk}")\n')
protect_new = (protect_old
               + '        lines.append(f"blk.{li}.attn_q.weight={attn_vk}")\n'
               + '        lines.append(f"blk.{li}.attn_output.weight={attn_vk}")\n')
if protect_old not in src:
    sys.exit("FATAL: attn_v/attn_k protection emit lines not found")
src = src.replace(protect_old, protect_new)

if src == orig:
    sys.exit("FATAL: no change applied")
pathlib.Path(str(p) + ".bak_attnfix").write_text(orig)
p.write_text(src)
print("PATCHED:", p)
print("  - removed attn_q/attn_output from BLOCK_TENSOR_ROLES")
print("  - added attn_q/attn_output to explicit per-layer Q5_K protection")
