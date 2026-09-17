#!/usr/bin/env bash
# Download the HumanML3D (t2m) evaluator archive and keep the files LaCT-Motion
# uses for rewards and evaluation:
#   checkpoints/t2m/Comp_v6_KLD005/opt.txt
#   checkpoints/t2m/text_mot_match/model/finest.tar
#   checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy
# Source: Motion-Agent release (https://github.com/szqwu/Motion-Agent), the same
# archive used by UniMo. Unlike the upstream script, this never removes the
# checkpoints/ directory, so SFT and GRPO checkpoints stored there are kept.
# The KIT-ML evaluator archive (kit.zip) is not needed and is not downloaded.
#
# Run from anywhere; the script works inside the project root. If t2m.zip is
# already present in the project root (manual download), the download is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE=t2m.zip
URL="https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view"
FILES=(
  Comp_v6_KLD005/opt.txt
  text_mot_match/model/finest.tar
  VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy
  VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
)

if [ ! -f "$ARCHIVE" ]; then
  command -v gdown >/dev/null || { echo "gdown not found: pip install gdown" >&2; exit 1; }
  echo "Downloading t2m extractors"
  gdown --fuzzy "$URL" -O "$ARCHIVE"
fi

tmp=$(mktemp -d ./.t2m_extract.XXXXXX)
unzip -q "$ARCHIVE" -d "$tmp"
src=$(find "$tmp" -type d -name t2m | head -n 1)
[ -n "$src" ] || { echo "t2m directory not found in $ARCHIVE" >&2; exit 1; }
for f in "${FILES[@]}"; do
  [ -f "$src/$f" ] || { echo "$f not found in $ARCHIVE" >&2; exit 1; }
  mkdir -p "checkpoints/t2m/$(dirname "$f")"
  cp "$src/$f" "checkpoints/t2m/$f"
done
rm -rf "$tmp" "$ARCHIVE"

find checkpoints/t2m -type f | sort
echo "Downloading done!"
