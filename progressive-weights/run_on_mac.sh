#!/usr/bin/env bash
# One-shot runner for an Apple Silicon Mac:  bash progressive-weights/run_on_mac.sh [WORKDIR]
#
#   1. makes a private venv and installs mlx, mlx-lm, transformers, pyarrow
#   2. downloads the model (resumable) and the WikiText-2 test split
#   3. converts the model to the progressive format (4-bit base + refinement tile file)
#   4. runs the REAL engine (GPU base + tiles read from your SSD)  -> tokens/s, MB read/token, quality
#   5. optionally runs the contention benchmark: do SSD reads slow the GPU?   (needs a file larger than your RAM)
#
# env (all optional):
#   MODEL=Qwen/Qwen2.5-3B     any Qwen2.5 size (0.5B 1.5B 3B 7B ...) or a Llama-style model; default Qwen/Qwen2.5-0.5B
#   ENGINE_SUITE=all|plan|drift    all = 7 per-token configurations; plan = decide once from the prompt (+4-bit resident);
#                                  drift = text that CHANGES TOPIC half way, with plan refresh
#   ENGINE_T=256  ENGINE_PLAN_LEN=64  ENGINE_WINDOWS=2  ENGINE_DRIFT_AT=128
#   SKIP_CONTENTION=1        skip the 56 GB contention test      SIZE_GB=56   its file size (must exceed your RAM)
#   PYTHON=python3.12        if your default python is too new for the mlx wheels
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HOME/pw_mac_run}"
PYTHON="${PYTHON:-python3}"
SIZE_GB="${SIZE_GB:-56}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
if [ "$MODEL" = "Qwen/Qwen2.5-0.5B" ]; then NAME=qwen05; CONV="$WORK/converted"      # reuse what an earlier run made
else NAME="$(echo "${MODEL##*/}" | tr 'A-Z' 'a-z')"; CONV="$WORK/converted_$NAME"; fi
M="$WORK/models/$NAME"
mkdir -p "$M" "$WORK/data"
echo "== work dir: $WORK   model: $MODEL   ($NAME)"

if [ ! -x "$WORK/venv/bin/python" ]; then
  echo "== creating venv with $($PYTHON --version 2>&1)"
  "$PYTHON" -m venv "$WORK/venv"
fi
"$WORK/venv/bin/pip" install --quiet --upgrade pip
if ! "$WORK/venv/bin/pip" install --quiet mlx mlx-lm transformers pyarrow numpy; then
  echo "!! pip could not install the packages. If the error mentions your Python version,"
  echo "   install an older Python (brew install python@3.12) and rerun with:  PYTHON=python3.12 bash $0"
  exit 1
fi
PY="$WORK/venv/bin/python"
"$PY" - <<'EOF'
import mlx.core as mx
print("== mlx default device:", mx.default_device())
assert "gpu" in str(mx.default_device()).lower(), "MLX is not using the GPU; this must run on an Apple Silicon Mac"
EOF

# complete = config.json plus every weight shard named in the index (or model.safetensors)
complete() { "$PY" - "$M" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
ok = os.path.exists(os.path.join(d, "config.json"))
idx = os.path.join(d, "model.safetensors.index.json")
if ok and os.path.exists(idx):
    ok = all(os.path.exists(os.path.join(d, f)) for f in set(json.load(open(idx))["weight_map"].values()))
else:
    ok = ok and os.path.exists(os.path.join(d, "model.safetensors"))
sys.exit(0 if ok else 1)
PYEOF
}
# The model comes from Hugging Face's own downloader: it resumes interrupted transfers and retries by itself
# (plain curl died with "HTTP/2 stream was not closed cleanly" on a slow connection).
if ! complete; then
  echo "== downloading $MODEL (resumable, up to 8 attempts; this can take a while for the bigger sizes)"
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if "$PY" - "$M" "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[2], local_dir=sys.argv[1], allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.tiktoken"])
PYEOF
    then ok=1; break; fi
    echo "   attempt $i failed, resuming in 5 s"; sleep 5
  done
  [ "$ok" = 1 ] || { echo "!! download failed 8 times; check your connection and rerun (it resumes)"; exit 1; }
  complete || { echo "!! download finished but the weight files are incomplete"; exit 1; }
fi
[ -s "$WORK/data/wikitext2_test.parquet" ] || { curl -fL --retry 5 -C - -# -o "$WORK/data/wikitext2_test.parquet.part" \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet" \
  && mv "$WORK/data/wikitext2_test.parquet.part" "$WORK/data/wikitext2_test.parquet"; }

cd "$HERE"
if [ ! -s "$CONV/members.npz" ]; then
  echo "== converting ($NAME)"
  PYTHONDONTWRITEBYTECODE=1 "$PY" mlx_engine.py convert --model "$M" --out "$CONV"
fi

SUITE="${ENGINE_SUITE:-all}"; ET="${ENGINE_T:-128}"; PL="${ENGINE_PLAN_LEN:-64}"; WN="${ENGINE_WINDOWS:-2}"; DA="${ENGINE_DRIFT_AT:-128}"
OUT="$WORK/engine_mac_${NAME}_${SUITE}"
echo
echo "== REAL ENGINE ($NAME, GPU base + tiles from the SSD), float16, suite=$SUITE, T=$ET, plan-len=$PL, windows=$WN"
PYTHONDONTWRITEBYTECODE=1 "$PY" mlx_engine.py bench --model "$M" --out "$CONV" \
  --data "$WORK/data/wikitext2_test.parquet" --dtype float16 --suite "$SUITE" --T "$ET" --plan-len "$PL" \
  --windows "$WN" --drift-at "$DA" --json "$OUT.json" 2>&1 | grep -v -i warning | tee "$OUT.txt"

if [ "${SKIP_CONTENTION:-0}" != "1" ]; then
  echo
  echo "== CONTENTION: SSD streaming vs GPU bandwidth   (creates a ${SIZE_GB} GB temp file, removed afterwards)"
  PYTHONDONTWRITEBYTECODE=1 "$PY" mac_contention_bench.py --size-gb "$SIZE_GB" --json "$WORK/contention_mac.json" 2>&1 | tee "$WORK/contention_mac.txt"
fi

echo
echo "== done. Paste me the output above, or the files in $WORK"
