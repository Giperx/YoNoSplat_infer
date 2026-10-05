#!/usr/bin/env bash
# One mode for every dataset.
#   bash metrics/run_all.sh single
#   bash metrics/run_all.sh multiframes
#   bash metrics/run_all.sh single --tag re10k --checkpoint pretrained_weights/re10k_224x224_ctx2to32.ckpt
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash metrics/run_all.sh single|multiframes [options]" >&2
  exit 1
fi
for name in nuscenes ddad lyft1920 lyft1224 widedrive; do
  bash "$DIR/run_wide.sh" "$name" "$@"
done
