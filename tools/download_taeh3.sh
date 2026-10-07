#!/usr/bin/env bash
# Download the TAEH3 tiny video decoder used by --taeh3.
#
# Pinned to madebyollin/taehv taeh3.safetensors (about 23 MiB).
# It decodes the same H3 latents as the full video VAE. Audio is unchanged.
#
# Usage:
#   ./tools/download_taeh3.sh
#   ./tools/download_taeh3.sh --dir ./FastH3-4-step-LoRA
#   ./tools/download_taeh3.sh --dry-run
set -euo pipefail

REV="62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
URL="https://raw.githubusercontent.com/madebyollin/taehv/${REV}/safetensors/taeh3.safetensors"
SHA256="4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
DEST="${H3_TAEH3_DIR:-./FastH3-4-step-LoRA}"
DRY=0

usage() {
    sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            shift
            [[ $# -gt 0 ]] || { echo "download_taeh3: --dir needs a path" >&2; exit 2; }
            DEST="$1"
            ;;
        --dry-run) DRY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "download_taeh3: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

OUT="$DEST/taeh3.safetensors"
echo "url: $URL"
echo "out: $OUT"

if [[ "$DRY" -eq 1 ]]; then
    echo "dry-run: not downloading"
    exit 0
fi

mkdir -p "$DEST"
curl -fsSL --retry 3 -o "$OUT" "$URL"
got=$(sha256sum "$OUT" | awk '{print $1}')
if [[ "$got" != "$SHA256" ]]; then
    echo "download_taeh3: sha256 mismatch" >&2
    echo "  expected $SHA256" >&2
    echo "  got      $got" >&2
    exit 1
fi
echo "taeh3 ready: $OUT"
echo "  ./h3 --taeh3 \"$OUT\" --fasth3-lora LORA -d \"\$H3_MODEL\" -p PROMPT --width 832 --height 480 --seconds 5 --layers 50 --reuse 1"
