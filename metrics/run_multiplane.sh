#!/usr/bin/env bash
# Multiplane inference, then photometric / HM, for one YoNoSplat dataset.
#
#   conda activate yonosplat
#   bash metrics/run_multiplane.sh nuscenes single
#   bash metrics/run_multiplane.sh ddad multiframes --max-frames 2
#
# The panorama is three predicted-pose planes, Left | Center | Right.
# Default context is 224x224. The stitched image is then resized to the same
# canvas as the single-pinhole wide render: 1176, 798, or 1050 wide by 224.
# Do not pass --output-dir. Results go to:
#   outputs/<dataset>_wide_pred_multiplane
#   outputs/<dataset>_wide_pred_multiframes_multiplane
#
# Ground truth is sparseMultiplaneImages3 at the saved canvas: 1176x224,
# 798x224, or 1050x224. Generate it with gen_wide_gt/multiPlanesWIdeFOV.
# Photometric and histogram matching are skipped until that directory exists.
# CBSR and PD stay disabled.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash metrics/run_multiplane.sh DATASET MODE [options] [inference flags...]

DATASET:
  nuscenes | ddad | lyft1920 | lyft1224 | widedrive

MODE:
  single         outputs/<dataset>_wide_pred_multiplane
  multiframes    outputs/<dataset>_wide_pred_multiframes_multiplane

Options:
  --skip-infer         score existing renders only
  --keep-aspect        each plane keeps the native aspect at height 224
  --tag NAME           append _NAME to the output directory
  --checkpoint PATH    inference checkpoint; default is the 224 dl3dv weight
  --gt-root PATH       override the sparseMultiplaneImages3 directory
  --val-list PATH      override the scene list

Other flags go to the inference script, for example --max-frames 1.
EOF
}

default_gt() {
  # Same saved canvas as the single-pinhole wide render.
  case "$1" in
    nuscenes) printf '%s' "datasets/nuscenes/sparseMultiplaneImages3_1176x224" ;;
    lyft1920) printf '%s' "datasets/lyft/1920_sparseMultiplaneImages3_1176x224" ;;
    lyft1224) printf '%s' "datasets/lyft/1224_sparseMultiplaneImages3_798x224" ;;
    ddad) printf '%s' "datasets/ddad_process/sparseMultiplaneImages3_1050x224" ;;
    widedrive) printf '%s' "datasets/WideDrive_processed/sparseMultiplaneImages3_1176x224" ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

take_value() {
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
    echo "missing value for ${1}" >&2
    exit 1
  fi
  printf '%s' "$2"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="$1"
shift
case "$DATASET" in
  nuscenes|ddad|lyft1920|lyft1224|widedrive) ;;
  *) echo "unknown dataset: $DATASET" >&2; usage >&2; exit 1 ;;
esac

if [[ $# -lt 1 ]]; then
  echo "missing mode: single or multiframes" >&2
  exit 1
fi
MODE="$1"
shift
case "$MODE" in
  single|multiframes) ;;
  *) echo "mode must be single or multiframes, got: $MODE" >&2; exit 1 ;;
esac

SKIP_INFER=0
KEEP_ASPECT=0
TAG=""
CHECKPOINT=""
GT_ROOT=""
VAL_LIST=""
INFER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skip-infer) SKIP_INFER=1; shift ;;
    --keep-aspect) KEEP_ASPECT=1; shift ;;
    --tag) TAG="$(take_value "$1" "${2:-}")"; shift 2 ;;
    --tag=*) TAG="${1#*=}"; shift ;;
    --checkpoint) CHECKPOINT="$(take_value "$1" "${2:-}")"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --gt-root) GT_ROOT="$(take_value "$1" "${2:-}")"; shift 2 ;;
    --gt-root=*) GT_ROOT="${1#*=}"; shift ;;
    --val-list) VAL_LIST="$(take_value "$1" "${2:-}")"; shift 2 ;;
    --val-list=*) VAL_LIST="${1#*=}"; shift ;;
    --output-dir|--output-dir=*)
      echo "Do not pass --output-dir." >&2
      exit 1
      ;;
    *) INFER_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$GT_ROOT" ]]; then
  GT_ROOT="$(default_gt "$DATASET")"
fi
if [[ "$KEEP_ASPECT" -eq 1 && -z "${GT_ROOT_SET:-}" ]]; then
  echo "keep-aspect changes the panorama width. Pass --gt-root for that canvas." >&2
fi

if [[ "$MODE" == "single" ]]; then
  RENDER_ROOT="outputs/${DATASET}_wide_pred_multiplane"
else
  RENDER_ROOT="outputs/${DATASET}_wide_pred_multiframes_multiplane"
  INFER_ARGS+=(--multiframes)
fi
if [[ "$KEEP_ASPECT" -eq 1 ]]; then
  RENDER_ROOT="${RENDER_ROOT}_aspect"
  INFER_ARGS+=(--keep-aspect)
fi
if [[ -n "$TAG" ]]; then
  if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "tag must be letters, numbers, dot, underscore, or dash: $TAG" >&2
    exit 1
  fi
  RENDER_ROOT="${RENDER_ROOT}_${TAG}"
fi
if [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

run_metrics() {
  local val_args=()
  if [[ -n "$VAL_LIST" ]]; then
    val_args=(--val-list "$VAL_LIST")
  fi
  if [[ -d "$GT_ROOT" ]]; then
    echo "Photometric -> $RENDER_ROOT"
    python metrics/eval_photometric.py --dataset "$DATASET" --mode "$MODE" \
      --render-root "$RENDER_ROOT" --gt-root "$GT_ROOT" "${val_args[@]}"
    echo "Histogram match -> $RENDER_ROOT"
    python metrics/eval_photometric.py --dataset "$DATASET" --mode "$MODE" \
      --render-root "$RENDER_ROOT" --gt-root "$GT_ROOT" --histogram-match "${val_args[@]}"
  else
    echo "GT root missing, skip photometric/HM: $GT_ROOT" >&2
  fi
  if [[ "$DATASET" == "lyft1920" || "$DATASET" == "lyft1224" ]]; then
    python metrics/merge_lyft.py --render-root "$RENDER_ROOT"
  fi
}

echo "Mode: $MODE -> $RENDER_ROOT"
if [[ -n "$CHECKPOINT" ]]; then
  echo "Checkpoint: $CHECKPOINT"
fi
if [[ "$SKIP_INFER" -eq 0 ]]; then
  launch=(python scripts/inference_nuscenes_wide_pred_multiplane.py --dataset "$DATASET" --output-dir "$RENDER_ROOT")
  if [[ -n "$CHECKPOINT" ]]; then
    launch+=(--checkpoint "$CHECKPOINT")
  fi
  if ((${#INFER_ARGS[@]})); then
    launch+=("${INFER_ARGS[@]}")
  fi
  "${launch[@]}"
fi
run_metrics
