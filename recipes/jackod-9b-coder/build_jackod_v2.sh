#!/usr/bin/env bash
# JackOD-9B-Coder v2 (re-release 2026-10-21): the JackOD recipe with Jackrong's Coder-Flash in the 0.30 slot that
# held Qwopus3.5-9B-Coder in v1. Delta .55 / Coder-Flash .30 / Ornith .15, omnimerge_v2, base == task-base ==
# Qwen3.5-9B. This is the merge + patch half of the build that produced the published weights (the eval/quant
# half is bs2_jackod_v2_ac_imatrix.sh + bs2_jackod_v2_sweep.sh).
#
#   WORK=/path/to/workdir OMK=/path/to/omnimergekit bash build_jackod_v2.sh
#
# Inputs are pinned by revision. DeltaCoder ships as a LoRA (root adapter, r32/a32 -> scale 1.0); it is applied
# to dense with scripts/apply_lora_to_safetensors.py, never through a transformers load/save round-trip (that
# drops mtp.* and nests key prefixes).
set -uo pipefail
W="${WORK:?WORK must point at a work dir with ~170 GB free}"
OMK="${OMK:?OMK must point at an omnimergekit checkout}"
PY="${PY:-python}"
M=$W/models; D=$M/DeltaCoder-9B-applied; O=$W/JackOD-9B-Coder
fail(){ echo "JACKOD_V2_FAIL $*"; exit 1; }
mkdir -p "$M"

dl(){ [ -f "$3/.dl_ok" ] && return 0
      hf download "$1" --revision "$2" --local-dir "$3" "${@:4}" || fail "download $1"; touch "$3/.dl_ok"; }
dl Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a "$M/Qwen3.5-9B"
dl Jackrong/Qwopus3-5-9B-Coder-Flash a972c214613d831b6c03b4665c0c97f2a3f90848 "$M/Qwopus3-5-9B-Coder-Flash" --exclude "assets/*"
dl ornith-ai/Ornith-1.5-9B 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 "$M/Ornith-1.5-9B"
# filenames positional: `hf download --include a b` keeps only the LAST pattern
dl danielcherubini/Qwen3.5-DeltaCoder-9B 10a2546fff9d80a418dfd883fa5f3a5c63bbbbe7 "$M/deltacoder" adapter_config.json adapter_model.safetensors
for d in Qwen3.5-9B Qwopus3-5-9B-Coder-Flash Ornith-1.5-9B; do  # a zero exit is not the artifact
  for f in config.json model.safetensors.index.json tokenizer.json; do [ -f "$M/$d/$f" ] || fail "$d lacks $f"; done
done
for f in adapter_config.json adapter_model.safetensors; do [ -f "$M/deltacoder/$f" ] || fail "deltacoder lacks $f"; done

[ -f "$D/.apply_ok" ] || { "$PY" "$OMK/scripts/apply_lora_to_safetensors.py" "$M/Qwen3.5-9B" "$M/deltacoder" "$D" \
                             || fail "apply_lora"; touch "$D/.apply_ok"; }
"$PY" census_jackod.py "$M/Qwen3.5-9B" "$D" || fail "DeltaCoder-applied census"

"$PY" "$OMK/omnimergekit.py" \
  --base "$M/Qwen3.5-9B" --task-base "$M/Qwen3.5-9B" \
  --source "$D" --source "$M/Qwopus3-5-9B-Coder-Flash" --source "$M/Ornith-1.5-9B" \
  --weights 0.55,0.30,0.15 \
  --method omnimerge_v2 --density 0.53 --darex-q 0.75 --seed 42 --no-auto-mlp-skip --skip-patterns visual.,mtp. \
  --output "$O" || fail "merge"

# Root-eos patch. Coder-Flash, like Qwen3.5-9B and DeltaCoder-applied, has NO root eos_token_id, so v1's
# patch_jackod.py (copy root eos from the Qwopus-slot source) would copy nothing: force Qwopus' values.
"$PY" patch_jackod.py "$M/Qwopus3-5-9B-Coder-Flash" "$O" || fail "tokenizer/template patch"
"$PY" - "$O" <<'PYP' || fail "root eos"
import json, os, sys
p = os.path.join(sys.argv[1], "config.json")
c = json.load(open(p)); c["eos_token_id"], c["pad_token_id"] = 248046, 248055
json.dump(c, open(p, "w"), indent=2); print("  root eos 248046 / pad 248055")
PYP
"$PY" census_jackod.py "$M/Qwen3.5-9B" "$O" || fail "merge census"
echo "JACKOD_V2_BUILD_DONE $O"
