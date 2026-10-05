#!/usr/bin/env bash
# Lyft 1920 and 1224 are one dataset. After metrics, run_wide.sh pools both
# subsets by frame count into outputs/lyft1920_*/yonosplat_lyft.txt.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_wide.sh" lyft1920 "$@"
