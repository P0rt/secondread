#!/usr/bin/env bash
# Export a downloaded Tinker LoRA checkpoint to a llama.cpp LoRA GGUF for SecondRead.
#
#   uv run python scripts/fetch_adapter.py tinker://<run-id>/weights/final --out ckpt
#   scripts/export_adapter.sh ckpt/<new-run-id>_external_weights_final    # or just: ckpt
#
# fetch_adapter.py uses save_weights_external + get_external_weights_urls. The older
#   tinker checkpoint download tinker://<run-id>/sampler_weights/final --output ./ckpt
# (documented to extract into ckpt/<run-id>_sampler_weights_final/) is still accepted, but
# its archive endpoint is deprecated in tinker 0.33.1 and hung for >10 min on 2026-10-14,
# so that layout was never seen with a real checkpoint.
#
# Pipeline (no base weights are ever downloaded; HF_HUB_OFFLINE=1 throughout):
#   1. validate the Tinker checkpoint and convert it to PEFT with
#      tinker_cookbook.weights.build_lora_adapter against the header-only stub base35/
#   2. tools/llama.cpp/convert_lora_to_gguf.py (tag b11429 = the brew llama-server build)
#   3. replay llama.cpp's adapter-loader name/shape checks against the base GGUF offline
#   4. load base GGUF + LoRA in a throwaway llama-server and check that P(Major) moves
#      between LoRA scale 0 and 1 (skip with --no-verify)
# The output is written only after every step passed, so a failed run never leaves a
# half-valid models/secondread-lora.gguf behind.
#
# usage: scripts/export_adapter.sh [--no-verify] [--port N] [--force] CKPT_DIR [OUT_GGUF]
#   CKPT_DIR   dir with adapter_model.safetensors + adapter_config.json, or a parent of
#              exactly one such dir (e.g. ./ckpt)
#   OUT_GGUF   default models/secondread-lora.gguf
#   --port N   port for the verification llama-server (default 9932; must be free)
#   --force    overwrite OUT_GGUF if it exists
#   env OUTTYPE=f16|bf16|f32|q8_0 (default f16, as verified)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LLAMA_TAG="b11429"                         # matches `llama-server --version`: build 11429, commit d81235049
LLAMA_DIR="tools/llama.cpp"
CONVERTER="$LLAMA_DIR/convert_lora_to_gguf.py"
BASE_STUB="base35"
BASE_GGUF="models/Qwen3.5-4B-Q4_K_M.gguf"
OUTTYPE="${OUTTYPE:-f16}"

VERIFY=1
PORT=9932
FORCE=0
POSITIONAL=()

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
step() { printf '\n== %s\n' "$*"; }
usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "$0"; exit "${1:-0}"; }

while (($#)); do
  case "$1" in
    --no-verify) VERIFY=0 ;;
    --port) [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || die "--port needs a number"; PORT="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) usage 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done
((${#POSITIONAL[@]} >= 1 && ${#POSITIONAL[@]} <= 2)) || usage 2
CKPT="${POSITIONAL[0]}"
OUT="${POSITIONAL[1]:-models/secondread-lora.gguf}"

# Never let any HF code path fall back to snapshot_download of the 9.3 GB bf16 repo.
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1  # keep python stdout/stderr in order in the log
PY=(uv run --frozen python)  # --frozen: never touch uv.lock

# ---------------------------------------------------------------------------
step "preflight"

command -v uv >/dev/null || die "uv not found (the project env is run with 'uv run')"

[[ -f "$CONVERTER" ]] || die "llama.cpp converter not found at $CONVERTER. Clone it:
  git clone --depth 1 --branch $LLAMA_TAG --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp $LLAMA_DIR
  git -C $LLAMA_DIR sparse-checkout set conversion gguf-py requirements"

# The converter and the runtime must agree on tensor layout: compare the clone's commit
# with the commit llama-server was built from.
clone_commit="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
if command -v llama-server >/dev/null; then
  server_commit="$(llama-server --version 2>&1 | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -1)"
  if [[ -n "$server_commit" && "$clone_commit" != "$server_commit"* ]]; then
    warn "converter clone is at ${clone_commit:0:9} but llama-server was built from $server_commit;
         re-clone $LLAMA_DIR at the tag matching 'llama-server --version' (bNNNNN) if loading fails"
  fi
  echo "llama.cpp: converter ${clone_commit:0:9}, llama-server ${server_commit:-unknown}"
elif ((VERIFY)); then
  die "llama-server not found (brew install llama.cpp), or pass --no-verify"
fi

stub_hint="rebuild the header-only stub (172 KB on disk, never copy it without sparse support):
  mkdir -p $BASE_STUB && for f in config.json model.safetensors.index.json; do curl -fL -o $BASE_STUB/\$f https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/\$f; done
  python3 scripts/mkstub.py Qwen/Qwen3.5-4B $BASE_STUB model.safetensors-00001-of-00002.safetensors=5329398688 model.safetensors-00002-of-00002.safetensors=3990429408"
for f in config.json model.safetensors.index.json \
         model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors; do
  [[ -s "$BASE_STUB/$f" ]] || die "$BASE_STUB/$f is missing; $stub_hint"
done
[[ -f "$BASE_GGUF" ]] || die "$BASE_GGUF not found:
  curl -L -o $BASE_GGUF https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# Locate the adapter. `tinker checkpoint download` extracts into
# ./ckpt/<run-id>_sampler_weights_final/ and deletes the tar, so accept that dir or its parent.
[[ "$CKPT" == tinker://* ]] && die "$CKPT is a Tinker path; download it first:
  uv run python scripts/fetch_adapter.py tinker://<run-id>/weights/<name> --out ./ckpt"
[[ -d "$CKPT" ]] || die "checkpoint dir $CKPT does not exist"
if [[ -f "$CKPT/adapter_model.safetensors" ]]; then
  ADAPTER="$CKPT"
else
  found=()
  while IFS= read -r f; do found+=("$(dirname "$f")"); done \
    < <(find "$CKPT" -maxdepth 3 -name adapter_model.safetensors 2>/dev/null | sort)
  if ((${#found[@]} == 0)); then
    tars="$(find "$CKPT" -maxdepth 2 -name '*.tar*' 2>/dev/null | head -3)"
    [[ -n "$tars" ]] && die "no extracted adapter under $CKPT, only archive(s):
$tars
  extract with: tar -xf <archive> -C $CKPT"
    die "no adapter_model.safetensors under $CKPT (expected ckpt/<run-id>_external_weights_final/ or ckpt/<run-id>_sampler_weights_final/).
  Download the adapter files with:
  uv run python scripts/fetch_adapter.py tinker://<run-id>/weights/final --out ./ckpt"
  fi
  ((${#found[@]} == 1)) || die "$CKPT holds ${#found[@]} adapters, pass one of them:
$(printf '  %s\n' "${found[@]}")"
  ADAPTER="${found[0]}"
fi
[[ -f "$ADAPTER/adapter_config.json" ]] || die "$ADAPTER has adapter_model.safetensors but no adapter_config.json"
echo "adapter: $ADAPTER"

[[ -e "$OUT" && $FORCE -eq 0 ]] && die "$OUT already exists (use --force to overwrite)"
mkdir -p "$(dirname "$OUT")"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/secondread-export.XXXXXX")"
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
step "1/4 Tinker checkpoint -> PEFT (build_lora_adapter, base=$BASE_STUB)"
"${PY[@]}" scripts/adapter_export.py to-peft "$ADAPTER" "$TMP/peft" --base "$BASE_STUB"

# ---------------------------------------------------------------------------
step "2/4 PEFT -> LoRA GGUF ($CONVERTER, outtype=$OUTTYPE)"
if ! "${PY[@]}" "$CONVERTER" --base "$BASE_STUB" --outtype "$OUTTYPE" \
       --outfile "$TMP/lora.gguf" "$TMP/peft" >"$TMP/convert.log" 2>&1; then
  grep -v '^INFO:hf-to-gguf:blk\.' "$TMP/convert.log" | tail -20 >&2
  die "convert_lora_to_gguf.py failed (log above). 'Can not map tensor' means a module llama.cpp
  does not know under that name (e.g. split in_proj_q/k/v: retrain with train_attn=False)"
fi
grep -E 'n_tensors|successfully' "$TMP/convert.log" | sed 's/^INFO:[a-z_.-]*://'

# ---------------------------------------------------------------------------
step "3/4 offline loader checks against $BASE_GGUF"
"${PY[@]}" scripts/adapter_export.py check-gguf "$TMP/lora.gguf" --peft "$TMP/peft" --base-gguf "$BASE_GGUF"

# ---------------------------------------------------------------------------
if ((VERIFY)); then
  step "4/4 llama-server load + LoRA effect (port $PORT)"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 ||
     curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    die "port $PORT is already in use; pick a free one with --port (9931 is the app's server)"
  fi
  # -lv 4 makes libllama log the adapter load ("loaded N tensors from lora file").
  llama-server -m "$BASE_GGUF" --lora "$TMP/lora.gguf" -c 4096 -np 1 \
    --host 127.0.0.1 --port "$PORT" -lv 4 >"$TMP/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      grep -iE 'lora|adapter|error|failed' "$TMP/server.log" | tail -15 >&2
      SERVER_PID=""
      die "llama-server exited while loading the adapter (log above); the GGUF was not written"
    fi
    sleep 1
  done
  curl -sf "http://127.0.0.1:$PORT/health" | grep -q ok || die "llama-server did not become healthy in 120 s"
  loaded="$(grep -o 'loaded [0-9]* tensors from lora file' "$TMP/server.log" || true)"
  [[ -n "$loaded" ]] || die "llama-server is up but did not report loading the LoRA"
  echo "llama-server: $loaded"
  "${PY[@]}" scripts/adapter_export.py smoke --port "$PORT"
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
else
  step "4/4 skipped (--no-verify)"
fi

mv -f "$TMP/lora.gguf" "$OUT"
step "done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "run: llama-server -m $BASE_GGUF --lora $OUT -c 16384 -np 4 -cram 0 --host 127.0.0.1 --port 9931"
echo "per-request scale: {\"lora\":[{\"id\":0,\"scale\":0}]} gives the base model for comparison"
