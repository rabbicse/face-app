#!/usr/bin/env bash
# Puts w600k_mbf.onnx (MobileFaceNet ArcFace, 512-d) into assets/models/.
# It is the *same file* the web app uses, so phones and browsers share one embedding space.
# InsightFace pretrained models are released for non-commercial research use only.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=assets/models/w600k_mbf.onnx
SHA256=9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f
WEB_COPY=../../frontend/frs_react/public/models/arcface/w600k_mbf.onnx
URL=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip

mkdir -p assets/models
if [[ -f "$DEST" ]]; then
  echo "$DEST already present"
elif [[ -f "$WEB_COPY" ]]; then
  cp "$WEB_COPY" "$DEST"
  echo "copied from the web app"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  echo "downloading $URL"
  curl -fL --progress-bar -o "$tmp/buffalo_s.zip" "$URL"
  unzip -q -j "$tmp/buffalo_s.zip" "*w600k_mbf.onnx" -d "$tmp"
  mv "$tmp/w600k_mbf.onnx" "$DEST"
fi

actual=$( (sha256sum "$DEST" 2>/dev/null || shasum -a 256 "$DEST") | cut -d' ' -f1)
if [[ "$actual" != "$SHA256" ]]; then
  echo "warning: $DEST differs from the InsightFace release (custom model?) - the web app, this app and" \
       "the backend (FRS_EMBEDDING_MODELS) must all use the same model" >&2
fi
