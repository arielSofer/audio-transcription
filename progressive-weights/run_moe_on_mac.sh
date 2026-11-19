#!/usr/bin/env bash
# Expert streaming on a Mac:  bash progressive-weights/run_moe_on_mac.sh [WORKDIR]
#
#   1. reuses the venv from run_on_mac.sh (creates it if missing), downloads a MoE model (resumable)
#   2. converts its experts into one 4 KB-aligned file next to the model (about the size of the model)
#   3. runs the normal model with every expert resident (speed + quality baseline; skipped for models > RAM),
#      then the STREAMED model with a few cache sizes: same perplexity, tokens/s, experts read per token, SSD MB/token
#   4. records which experts each token used and prints how well a cache or a "prompt-fixed" set would hit
#
# env (optional):
#   MODEL=mlx-community/Qwen3-30B-A3B-4bit   (17 GB)   or mlx-community/gpt-oss-20b-MXFP4-Q8 (12 GB),
#                                            or mlx-community/gpt-oss-120b-MXFP4-Q8 (63 GB, does not fit RAM -> VERIFY=0)
#   CACHE_FRACS="0.125 0.25 0.5 1.0"   share of each layer's experts kept in RAM
#   VERIFY=1|0     also load the resident model (needs the whole model in memory); default 1
#   ENGINE_T=256  ENGINE_WINDOWS=2  ENGINE_START=64
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HOME/pw_mac_run}"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-mlx-community/Qwen3-30B-A3B-4bit}"
FRACS="${CACHE_FRACS:-0.125 0.25 0.5 1.0}"
VERIFY="${VERIFY:-1}"
T="${ENGINE_T:-256}"; WN="${ENGINE_WINDOWS:-2}"; ST="${ENGINE_START:-64}"
NAME="$(echo "${MODEL##*/}" | tr 'A-Z' 'a-z')"
M="$WORK/models/$NAME"; EXP="$WORK/experts_$NAME.bin"; OUT="$WORK/moe_${NAME}"
mkdir -p "$M" "$WORK/data"
echo "== work dir: $WORK   model: $MODEL"

if [ ! -x "$WORK/venv/bin/python" ]; then "$PYTHON" -m venv "$WORK/venv"; fi
"$WORK/venv/bin/pip" install --quiet --upgrade pip
"$WORK/venv/bin/pip" install --quiet mlx mlx-lm transformers pyarrow numpy
PY="$WORK/venv/bin/python"
"$PY" -c "import mlx.core as mx; print('== mlx default device:', mx.default_device())"

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
if ! complete || [ ! -s "$EXP.json" ]; then
  NEED="${NEED_GB:-35}"                                  # the download plus the expert file (about the same size again)
  FREE="$(df -k "$WORK" | awk 'NR==2 {printf "%d", $4/1048576}')"
  if [ "$FREE" -lt "$NEED" ]; then
    echo "!! only ${FREE} GB free in $WORK, this model needs about ${NEED} GB (the download plus the expert file)."
    echo "   free space (rm -rf the models or experts_*.bin you no longer need under $WORK) or rerun with NEED_GB=<smaller> if part is already on disk."
    exit 1
  fi
fi
if ! complete; then
  echo "== downloading $MODEL (resumable; this is large, be patient)"
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if "$PY" - "$M" "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[2], local_dir=sys.argv[1], allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.tiktoken", "*.jinja"])
PYEOF
    then ok=1; break; fi
    echo "   attempt $i failed, resuming in 5 s"; sleep 5
  done
  [ "$ok" = 1 ] || { echo "!! download failed 8 times; rerun (it resumes)"; exit 1; }
  complete || { echo "!! weights incomplete"; exit 1; }
fi
[ -s "$WORK/data/wikitext2_test.parquet" ] || curl -fL --retry 5 -C - -# -o "$WORK/data/wikitext2_test.parquet" \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"

cd "$HERE"
if [ ! -s "$EXP.json" ]; then
  echo "== converting the experts to $EXP"
  PYTHONDONTWRITEBYTECODE=1 "$PY" moe_stream.py convert --model "$M" --out "$EXP"
fi
read -r NE TK < <("$PY" - "$M" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
c = c.get("text_config", c)
ne = next(c[k] for k in ("num_experts", "num_local_experts", "n_routed_experts") if k in c)
tk = c.get("num_experts_per_tok", c.get("top_k", 8))
print(ne, tk)
PYEOF
)
echo "== $NE experts per layer, top-$TK"

first=1
for f in $FRACS; do
  extra=""
  if [ "$first" = 1 ]; then
    [ "$VERIFY" = 1 ] && extra="--verify"
    extra="$extra --trace $OUT.trace.json"
  fi
  echo
  echo "== STREAMED, cache = $f of the experts per layer  (T=$T, windows=$WN, scored from token $ST)  $([ -n "$extra" ] && echo "[$extra]")"
  PYTHONDONTWRITEBYTECODE=1 "$PY" moe_stream.py bench --model "$M" --experts "$EXP" \
    --data "$WORK/data/wikitext2_test.parquet" --T "$T" --windows "$WN" --start "$ST" --cache-frac "$f" $extra \
    --json "$OUT.cache_$f.json" 2>&1 | grep -v -i warning | tee -a "$OUT.txt"
  first=0
done

echo
echo "== EXPERT REUSE between tokens (from the recorded routing)"
"$PY" moe_reuse_analysis.py "$OUT.trace.json" --experts-per-layer "$NE" --top-k "$TK" --prompt "$ST" | tee -a "$OUT.txt"
echo
echo "== done. Paste me the output above, or $OUT.txt"
