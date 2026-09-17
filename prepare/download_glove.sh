#!/usr/bin/env bash
# Download the GloVe vocabulary used by the T2M evaluator and the GRPO rewards.
# Source: Motion-Agent release (https://github.com/szqwu/Motion-Agent), the same
# archive used by UniMo. Produces glove/our_vab_data.npy, our_vab_idx.pkl,
# and our_vab_words.pkl.
#
# Run from anywhere; the script works inside the project root. If glove.zip is
# already present in the project root (manual download), the download is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE=glove.zip
URL="https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing"

if [ ! -f "$ARCHIVE" ]; then
  command -v gdown >/dev/null || { echo "gdown not found: pip install gdown" >&2; exit 1; }
  echo "Downloading glove (used by the evaluators and rewards)"
  gdown --fuzzy "$URL" -O "$ARCHIVE"
fi

tmp=$(mktemp -d ./.glove_extract.XXXXXX)
unzip -q "$ARCHIVE" -d "$tmp"
mkdir -p glove
for f in our_vab_data.npy our_vab_idx.pkl our_vab_words.pkl; do
  src=$(find "$tmp" -type f -name "$f" | head -n 1)
  [ -n "$src" ] || { echo "$f not found in $ARCHIVE" >&2; exit 1; }
  cp "$src" "glove/$f"
done
rm -rf "$tmp" "$ARCHIVE"

ls -l glove
echo "Downloading done!"
