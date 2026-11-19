#!/usr/bin/env bash
# One-shot runner for an Apple Silicon Mac:  bash progressive-weights/run_on_mac.sh [WORKDIR]
#
#   1. makes a private venv and installs mlx, mlx-lm, transformers, pyarrow   (~2 GB of downloads incl. the model)
#   2. downloads Qwen2.5-0.5B and the WikiText-2 test split
#   3. converts the model to the progressive format (4-bit base + refinement tile file)
#   4. runs the REAL engine (GPU base + tiles read from your SSD) in 7 configurations -> tokens/s, MB read/token
#   5. runs the contention benchmark: do SSD reads slow the GPU?   (needs a test file larger than your RAM)
#
# env: PYTHON=python3.12   (use it if your default python is too new for the mlx wheels)
#      ENGINE_SUITE=plan ENGINE_T=256 SKIP_CONTENTION=1   (only the "decide once from the prompt" comparison, no 56 GB file)
#      SIZE_GB=56           (contention test file; must exceed your RAM, needs SIZE_GB+15 GB free disk)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HOME/pw_mac_run}"
PYTHON="${PYTHON:-python3}"
SIZE_GB="${SIZE_GB:-56}"
mkdir -p "$WORK/models/qwen05" "$WORK/data"
echo "== work dir: $WORK"

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

M="$WORK/models/qwen05"
# The model comes from Hugging Face's own downloader: it resumes interrupted transfers and retries by itself
# (plain curl died with "HTTP/2 stream was not closed cleanly" on a slow connection).
EXPECTED=988097824
have=0; [ -f "$M/model.safetensors" ] && have=$(wc -c < "$M/model.safetensors" | tr -d ' ')
if [ "$have" -lt "$EXPECTED" ]; then
  rm -f "$M/model.safetensors"                    # a partial file left by an older run would look complete
  echo "== downloading Qwen2.5-0.5B (about 1 GB; resumable, up to 8 attempts)"
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if "$PY" - "$M" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=sys.argv[1],
                  allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                                  "vocab.json", "merges.txt", "model.safetensors"])
PYEOF
    then ok=1; break; fi
    echo "   attempt $i failed, resuming in 5 s"; sleep 5
  done
  [ "$ok" = 1 ] || { echo "!! download failed 8 times; check your connection and rerun (it resumes)"; exit 1; }
fi
[ -s "$WORK/data/wikitext2_test.parquet" ] || { curl -fL --retry 5 -C - -# -o "$WORK/data/wikitext2_test.parquet.part" \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet" \
  && mv "$WORK/data/wikitext2_test.parquet.part" "$WORK/data/wikitext2_test.parquet"; }

cd "$HERE"
if [ ! -s "$WORK/converted/members.npz" ]; then
  echo "== converting"
  PYTHONDONTWRITEBYTECODE=1 "$PY" mlx_engine.py convert --model "$M" --out "$WORK/converted"
fi

SUITE="${ENGINE_SUITE:-all}"; ET="${ENGINE_T:-128}"
echo
echo "== REAL ENGINE (GPU base + tiles from the SSD), float16, suite=$SUITE, T=$ET.  One line per configuration."
PYTHONDONTWRITEBYTECODE=1 "$PY" mlx_engine.py bench --model "$M" --out "$WORK/converted" \
  --data "$WORK/data/wikitext2_test.parquet" --dtype float16 --suite "$SUITE" --T "$ET" \
  --json "$WORK/engine_mac_$SUITE.json" 2>&1 | grep -v -i warning | tee "$WORK/engine_mac_$SUITE.txt"

if [ "${SKIP_CONTENTION:-0}" != "1" ]; then
  echo
  echo "== CONTENTION: SSD streaming vs GPU bandwidth   (creates a ${SIZE_GB} GB temp file, removed afterwards)"
  PYTHONDONTWRITEBYTECODE=1 "$PY" mac_contention_bench.py --size-gb "$SIZE_GB" --json "$WORK/contention_mac.json" 2>&1 | tee "$WORK/contention_mac.txt"
fi

echo
echo "== done. Paste me the output above, or the files in $WORK"
