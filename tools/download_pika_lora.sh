#!/usr/bin/env bash
# Download a Pika H3 Flex LoRA used by --pika-lora / --pika-ref-lora.
#
# Repo: Pika-Labs/Pika-H3-Flex
# Default file: fl2va/adapter.safetensors, about 2.77 GB.
# --ref2va fetches ref2va/adapter.safetensors. --both fetches both.
# The base weights stay official MiniMax-H3 (FL2VA/ and, for references,
# Ref2VA/). Do not download this repo's copy of the base checkpoint.
#
# Usage:
#   ./tools/download_pika_lora.sh
#   ./tools/download_pika_lora.sh --ref2va --dir ./Pika-H3-Flex
#   ./tools/download_pika_lora.sh --both --dir ./Pika-H3-Flex
#   ./tools/download_pika_lora.sh --dry-run
#
# Needs the Hugging Face CLI (`hf` or `huggingface-cli`):
#   pip install -U "huggingface_hub[cli]"
# HF_TOKEN and HF_ENDPOINT are passed through to that CLI.
set -euo pipefail

REPO="${H3_PIKA_HF_REPO:-Pika-Labs/Pika-H3-Flex}"
DEST="${H3_PIKA_LORA_DIR:-./Pika-H3-Flex}"
MODE="fl2va"
DRY=0
FL2VA_SHA256="8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327"
REF2VA_SHA256="eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5"

usage() {
    sed -n '2,16p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            shift
            [[ $# -gt 0 ]] || { echo "download_pika_lora: --dir needs a path" >&2; exit 2; }
            DEST="$1"
            ;;
        --ref2va) MODE="ref2va" ;;
        --both) MODE="both" ;;
        --dry-run) DRY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "download_pika_lora: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

includes=()
case "$MODE" in
    fl2va)
        includes+=(fl2va/adapter.safetensors fl2va/manifest.json)
        ;;
    ref2va)
        includes+=(ref2va/adapter.safetensors ref2va/manifest.json)
        ;;
    both)
        includes+=(
            fl2va/adapter.safetensors fl2va/manifest.json
            ref2va/adapter.safetensors ref2va/manifest.json
        )
        ;;
esac

echo "repo:    $REPO"
echo "dir:     $DEST"
echo "include: ${includes[*]}"

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
    echo "download_pika_lora: need the Hugging Face CLI." >&2
    echo "  pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

mkdir -p "$DEST"
args=()
for item in "${includes[@]}"; do
    args+=(--include "$item")
done
"$cli" download "$REPO" --local-dir "$DEST" "${args[@]}"

check_sha() {
    local file="$1" expect="$2"
    if [[ ! -f "$file" ]]; then
        echo "download_pika_lora: missing $file" >&2
        exit 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "lora ready: $file (sha256sum not installed; not checked)"
        return 0
    fi
    local got
    got="$(sha256sum "$file" | awk '{print $1}')"
    if [[ "$got" != "$expect" ]]; then
        echo "download_pika_lora: sha256 mismatch for $file" >&2
        echo "  expected $expect" >&2
        echo "  got      $got" >&2
        exit 1
    fi
    echo "lora ready: $file"
    echo "  sha256 $got"
}

if [[ "$MODE" == "fl2va" || "$MODE" == "both" ]]; then
    check_sha "$DEST/fl2va/adapter.safetensors" "$FL2VA_SHA256"
    echo "  ./h3 --pika-lora \"$DEST/fl2va/adapter.safetensors\" -d \"\$H3_MODEL\" -p PROMPT --width 832 --height 480 --seconds 5 --layers 50 --reuse 1"
fi
if [[ "$MODE" == "ref2va" || "$MODE" == "both" ]]; then
    check_sha "$DEST/ref2va/adapter.safetensors" "$REF2VA_SHA256"
    echo "  ./h3 --pika-ref-lora \"$DEST/ref2va/adapter.safetensors\" --ref-image image.jpg -d \"\$H3_MODEL\" -p PROMPT --width 832 --height 480 --seconds 5 --layers 50 --reuse 1"
fi
