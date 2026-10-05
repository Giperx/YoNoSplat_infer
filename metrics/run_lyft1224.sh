#!/usr/bin/env bash
# Lyft 1224 is pooled with 1920 by frame count. The combined report is written
# under the matching lyft1920 output directory.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_wide.sh" lyft1224 "$@"
