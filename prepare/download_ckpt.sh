#!/usr/bin/env bash
# Download the Motion-Agent checkpoint archive and keep the motion VQ-VAE.
# Source: Motion-Agent release (https://github.com/szqwu/Motion-Agent), the same
# archive used by UniMo. Produces ckpt/vqvae.pth. The archive also contains
# motionllm.pth, which LaCT-Motion does not use; it is discarded.
#
# Run from anywhere; the script works inside the project root. If
# motion_agent.zip is already present in the project root (manual download),
# the download is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE=motion_agent.zip
URL="https://drive.google.com/file/d/1Tagt2xUwv_h0JNMtrM_Ty1rWemkLF5jH/view"

if [ ! -f "$ARCHIVE" ]; then
  command -v gdown >/dev/null || { echo "gdown not found: pip install gdown" >&2; exit 1; }
  echo "Downloading Motion-Agent ckpts"
  gdown --fuzzy "$URL" -O "$ARCHIVE"
fi

mkdir -p ckpt
# Extract only the VQ-VAE, ignoring the archive's internal directory name.
unzip -o -j "$ARCHIVE" '*vqvae.pth' -d ckpt
rm -f "$ARCHIVE"

ls -l ckpt/vqvae.pth
echo "Downloading done!"
