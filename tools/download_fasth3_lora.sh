#!/usr/bin/env bash
# Download the FastH3 dense 4-step LoRA used by --fasth3-lora.
#
# Repo: FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA
# File: dense-datafree/adapter_model.safetensors, about 1.4 GiB.
# The base weights stay official MiniMax-H3. This is not the VSA adapter.
#
# Usage:
#   ./tools/download_fasth3_lora.sh
#   ./tools/download_fasth3_lora.sh --dir ./FastH3-4-step-LoRA
#   ./tools/download_fasth3_lora.sh --dry-run
#
# Needs the Hugging Face CLI (`hf` or `huggingface-cli`):
#   pip install -U "huggingface_hub[cli]"
# HF_TOKEN and HF_ENDPOINT are passed through to that CLI.
set -euo pipefail

REPO="${H3_FASTH3_HF_REPO:-FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA}"
DEST="${H3_FASTH3_LORA_DIR:-./FastH3-4-step-LoRA}"
INCLUDE="dense-datafree/adapter_model.safetensors"
DRY=0

usage() {
    sed -n '2,14p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            shift
            [[ $# -gt 0 ]] || { echo "download_fasth3_lora: --dir needs a path" >&2; exit 2; }
            DEST="$1"
            ;;
        --dry-run) DRY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "download_fasth3_lora: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

echo "repo:    $REPO"
echo "dir:     $DEST"
echo "include: $INCLUDE"

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
    echo "download_fasth3_lora: need the Hugging Face CLI." >&2
    echo "  pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

mkdir -p "$DEST"
"$cli" download "$REPO" --local-dir "$DEST" --include "$INCLUDE"

lora="$DEST/$INCLUDE"
if [[ ! -f "$lora" ]]; then
    echo "download_fasth3_lora: missing $INCLUDE" >&2
    exit 1
fi

echo "lora ready: $lora"
echo "  ./h3 --fasth3-lora \"$lora\" -d \"\$H3_MODEL\" -p PROMPT --width 832 --height 480 --seconds 5 --layers 50 --reuse 1"
