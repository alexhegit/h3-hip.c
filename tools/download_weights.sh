#!/usr/bin/env bash
# Download only the MiniMax-H3 files h3-hip.c opens.
#
# The full Hugging Face repo is about 464 GiB. This tree loads FL2VA/ for
# T2VA and first/last-frame, and Ref2VA/ only when the request has ordered
# references. The root diffusers copies (transformer/, transformer_ref/,
# text_encoder/, vae/, audio_vae/) are not read.
#
# Usage:
#   ./tools/download_weights.sh
#   ./tools/download_weights.sh --dir "$H3_MODEL"
#   ./tools/download_weights.sh --ref2va          # add Ref2VA onto an existing tree
#   ./tools/download_weights.sh --both
#   ./tools/download_weights.sh --dry-run
#
# Needs the Hugging Face CLI (`hf` or `huggingface-cli`):
#   pip install -U "huggingface_hub[cli]"
# HF_TOKEN and HF_ENDPOINT are passed through to that CLI.
set -euo pipefail

REPO="${H3_HF_REPO:-MiniMaxAI/MiniMax-H3}"
DEST="${H3_MODEL:-./MiniMax-H3}"
MODE="fl2va"
DRY=0

usage() {
    sed -n '2,18p' "$0"
    cat <<'EOF'

Modes (pick one):
  --fl2va     T2VA and first/last-frame. About 134 GiB. Default.
  --ref2va    Reference-to-video pack only. About 134 GiB.
  --both      FL2VA plus Ref2VA. About 268 GiB.
              Still skips the unused root copies (about 196 GiB).

Options:
  --dir DIR   Destination directory (default: $H3_MODEL or ./MiniMax-H3).
  --dry-run   Print the include patterns and the files h3 checks. No download.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fl2va) MODE="fl2va" ;;
        --ref2va) MODE="ref2va" ;;
        --both) MODE="both" ;;
        --dir)
            shift
            [[ $# -gt 0 ]] || { echo "download_weights: --dir needs a path" >&2; exit 2; }
            DEST="$1"
            ;;
        --dry-run) DRY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "download_weights: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

partition_includes() {
    local part="$1"
    printf '%s\n' \
        "${part}/model_index.json" \
        "${part}/transformer/*" \
        "${part}/text_encoder/*" \
        "${part}/tokenizer/*" \
        "${part}/video_vae/*" \
        "${part}/video_vae/source/*" \
        "${part}/audio_vae/*"
}

includes=()
case "$MODE" in
    fl2va) mapfile -t includes < <(partition_includes FL2VA) ;;
    ref2va) mapfile -t includes < <(partition_includes Ref2VA) ;;
    both)
        mapfile -t includes < <(partition_includes FL2VA; partition_includes Ref2VA)
        ;;
esac

echo "repo:  $REPO"
echo "dir:   $DEST"
echo "mode:  $MODE"
echo "include:"
printf '  %s\n' "${includes[@]}"

if [[ "$DRY" -eq 1 ]]; then
    echo "dry-run: not downloading"
    exit 0
fi

cli=""
if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then
    cli="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    cli="huggingface-cli"
else
    echo "download_weights: need the Hugging Face CLI." >&2
    echo "  pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

mkdir -p "$DEST"
args=(download "$REPO" --local-dir "$DEST")
for pat in "${includes[@]}"; do
    args+=(--include "$pat")
done
"$cli" "${args[@]}"

missing=0
require() {
    if [[ ! -f "$DEST/$1" ]]; then
        echo "download_weights: missing $1" >&2
        missing=1
    fi
}

if [[ "$MODE" == "fl2va" || "$MODE" == "both" ]]; then
    require "FL2VA/transformer/config.json"
    require "FL2VA/transformer/model.safetensors.index.json"
    require "FL2VA/tokenizer/tokenizer.json"
    require "FL2VA/text_encoder/model.safetensors.index.json"
    require "FL2VA/video_vae/source/model.safetensors"
    require "FL2VA/audio_vae/model.safetensors"
fi
if [[ "$MODE" == "ref2va" || "$MODE" == "both" ]]; then
    require "Ref2VA/transformer/config.json"
    require "Ref2VA/transformer/model.safetensors.index.json"
    require "Ref2VA/tokenizer/tokenizer.json"
    require "Ref2VA/text_encoder/model.safetensors.index.json"
    require "Ref2VA/video_vae/source/model.safetensors"
    require "Ref2VA/audio_vae/model.safetensors"
fi
if [[ "$missing" -ne 0 ]]; then
    echo "download_weights: download finished but required files are missing" >&2
    exit 1
fi

echo "weights ready: $DEST"
echo "  export H3_MODEL=$DEST"
echo "  ./h3 --info -d \"\$H3_MODEL\""
