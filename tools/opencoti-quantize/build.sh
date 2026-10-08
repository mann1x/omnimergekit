#!/bin/bash
# build.sh <opencoti build tree> [out] — link the opencoti `llama_quantize` (PQ2_0 / PTQ1_0 capable) from an opencoti
# llamafile build tree that has already been built (opencoti >= 9658a3f530, i.e. patch 0590, built by its dev-build script).
# The opencoti pipeline compiles quantize.cpp but does not link a quantize binary, and BUILD.mk's own
# `o//llama.cpp/quantize/quantize` rule falls to an implicit link of one object, so the link is spelled out here.
# Example: ./build.sh <opencoti checkout>/vendors/sources/llamafile bin/opencoti-quantize
set -eu
T=$(cd "${1:?opencoti llamafile build tree (vendors/sources/llamafile)}" && pwd)
OUT=$(realpath -m "${2:-bin/opencoti-quantize}"); mkdir -p "$(dirname "$OUT")"
cd "$T"
MK=$(ls -d .cosmocc/*/bin/make | sort -V | tail -1); CXX=$(dirname "$MK")/cosmoc++
for f in o//llama.cpp/tools/quantize/quantize.cpp.o o//llama.cpp/llama.cpp.a o//llamafile/llamafile.o o//llamafile/zip.o o//llamafile/gpu.a; do
  [ -f "$f" ] || { echo "missing $f — build the tree first"; exit 2; }
done
grep -q "PTQ1_0" llama.cpp/tools/quantize/quantize.cpp || { echo "this tree's quantize has no PTQ1_0 (needs opencoti patch 0590)"; exit 2; }
TB=$("$MK" -pn o//llama.cpp/llama.cpp.a 2>/dev/null | grep -E "^TINYBLAS_CPU_OBJS :?=" | head -1 | sed -E 's/^TINYBLAS_CPU_OBJS :?= //')
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
"$CXX" -O2 -g -mclang -c llama.cpp/tools/quantize/main.cpp -o "$TMP/qmain.o"   # the 2-line main() calling llama_quantize()
"$CXX" -O2 -g -fexceptions -ffunction-sections -fdata-sections -mclang -fopenmp \
  "$TMP/qmain.o" o//llama.cpp/tools/quantize/quantize.cpp.o $TB o//llamafile/llamafile.o o//llamafile/zip.o \
  o//llama.cpp/llama.cpp.a o//llamafile/gpu.a -lpthread -o "$OUT"
"$OUT" --help 2>&1 | grep -E "PQ2_0|PTQ1_0"
sha256sum "$OUT"
