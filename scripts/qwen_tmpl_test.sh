#!/usr/bin/env bash
# Reproduce gsm8k few-shot regurgitation (default GGUF template) vs fix
# (--chat-template-file eval_fewshot.jinja). GPU0 debug server, MTP binary.
set -u
NEW=/srv/ml/repos/llama.cpp-latest/build/bin
GGUF=/mnt/sdc/ml/gguf/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
FIXED=/mnt/sdc/ml/google/qwen36-35b-a3b-tok/chat_template_eval_fewshot.jinja
PORT=8093
# 3-shot gsm8k (real fewshot) + a test question (Weng: 12*50/60 = $10)
REQ='{"messages":[
 {"role":"user","content":"Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nAnswer:"},
 {"role":"assistant","content":"Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"},
 {"role":"user","content":"Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer:"}
 ],"max_tokens":2048,"temperature":0,"stop":["</s>","<|im_end|>"]}'
run() {
  local LABEL="$1"; shift
  CUDA_VISIBLE_DEVICES=0 nohup "$NEW/llama-server" -m "$GGUF" --port $PORT -c 32768 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192 -ctk q8_0 -ctv q8_0 "$@" \
    > /srv/ml/logs/t177/tmpltest_${LABEL}.log 2>&1 < /dev/null & disown
  for i in $(seq 1 90); do curl -s localhost:$PORT/health 2>/dev/null | grep -q ok && break; sleep 2; done
  echo "[$LABEL] $(curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$REQ" \
     | python3 -c "import json,sys;d=json.load(sys.stdin);c=d[\"choices\"][0][\"message\"];print(\"content=\",repr((c.get(\"content\") or \"\")[:220]),\"| reason_len=\",len(c.get(\"reasoning_content\") or \"\"),\"| finish=\",d[\"choices\"][0].get(\"finish_reason\"))")"
  pkill -f "llama-server.*$PORT" 2>/dev/null; sleep 2
}
echo "=== A: DEFAULT GGUF template (reproduce bug) ==="
run default
echo "=== B: FIXED --chat-template-file (eval_fewshot) ==="
run fixed --chat-template-file "$FIXED"
