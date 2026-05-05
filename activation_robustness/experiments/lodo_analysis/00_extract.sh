#!/usr/bin/env bash
# Extract the canonical LODO results tarball into a stable location.
# Idempotent: wipes prior extracted/ contents first.
#
# Source of truth: ./copy_of_aws_8gpu_lodo_results_latest
# (the older non-_latest tarball is corrupted; do not use it.)
set -euo pipefail

TARBALL="./copy_of_aws_8gpu_lodo_results_latest"
EXTRACT_ROOT="./lodo_data/extracted"

if [[ ! -f "$TARBALL" ]]; then
  echo "ERROR: tarball not found at $TARBALL" >&2
  exit 1
fi

echo "[00_extract] Tarball: $TARBALL"
echo "[00_extract] Tarball size: $(stat -c '%s' "$TARBALL") bytes"
echo "[00_extract] Tarball mtime: $(stat -c '%y' "$TARBALL")"

echo "[00_extract] Wiping $EXTRACT_ROOT"
rm -rf "$EXTRACT_ROOT"
mkdir -p "$EXTRACT_ROOT"

echo "[00_extract] Extracting..."
tar -xzf "$TARBALL" -C "$EXTRACT_ROOT"

echo "[00_extract] Top-level contents:"
ls -1 "$EXTRACT_ROOT/lodo_results_export/" | head -10

echo "[00_extract] Done. Export root: $EXTRACT_ROOT/lodo_results_export"
