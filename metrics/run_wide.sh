#!/usr/bin/env bash
# Inference, then photometric / HM / CBSR / PD, for one YoNoSplat dataset.
#
#   conda activate yonosplat
#   bash metrics/run_wide.sh nuscenes single
#   bash metrics/run_wide.sh ddad multiframes --max-frames 2
#   bash metrics/run_lyft1920.sh single --skip-infer
#
# Do not pass --output-dir. Metrics read:
#   outputs/<dataset>_wide_pred
#   outputs/<dataset>_wide_pred_multiframes
# --keep-aspect writes the matching *_aspect directory instead. The saved
# image size does not change; only the encoder input keeps its aspect ratio.
#
# CBSR and PD do not use GT. Photometric and histogram matching are skipped
# until that GT directory exists. eval_crcs.py and eval_ips.py are not called.
# After the dataset finishes, per-scene match/ and matched_img/ are removed.
#
# FPS is separate and times the whole forward, including every frame of a
# multi-frame window:
#   python scripts/benchmark_nuscenes.py
#   python scripts/benchmark_nuscenes_multiframes.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash metrics/run_wide.sh DATASET MODE [options] [inference flags...]

DATASET:
  nuscenes | ddad | lyft1920 | lyft1224 | widedrive

MODE:
  single         outputs/<dataset>_wide_pred
  multiframes    outputs/<dataset>_wide_pred_multiframes

Options:
  --skip-infer         score existing renders only
  --keep-aspect        infer at height 224 with the native aspect
                       and write outputs/<dataset>_wide_pred[_multiframes]_aspect
  --tag NAME           write outputs/<dataset>_wide_pred[_multiframes]_NAME
  --checkpoint PATH    inference checkpoint; default is the 224 dl3dv weight
  --gt-root PATH       override the height-224 sparse GT directory
  --val-list PATH      override the scene list

Other flags go to the inference script, for example --max-frames 1 or --scene 000.
EOF
}

default_gt() {
  case "$1" in
    nuscenes) printf '%s' "datasets/nuscenes/sparseWideFOVImages3_1176x224" ;;
    lyft1920) printf '%s' "datasets/lyft/1920_sparseWideFOVImages3_1176x224" ;;
    lyft1224) printf '%s' "datasets/lyft/1224_sparseWideFOVImages3_798x224" ;;
    ddad) printf '%s' "datasets/ddad_process/sparseWideFOVImages3_1050x224" ;;
    widedrive) printf '%s' "datasets/WideDrive_processed/sparseWideFOVImages3_1176x224" ;;
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

if [[ "$MODE" == "single" ]]; then
  INFER_SCRIPT="scripts/inference_nuscenes_wide_pred.py"
  RENDER_ROOT="outputs/${DATASET}_wide_pred"
else
  INFER_SCRIPT="scripts/inference_nuscenes_wide_pred_multiframes.py"
  RENDER_ROOT="outputs/${DATASET}_wide_pred_multiframes"
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

cleanup_match_dirs() {
  local root="$1" scene_dir name target removed=0
  [[ -d "$root" ]] || return 0
  for scene_dir in "$root"/*/; do
    [[ -d "$scene_dir" ]] || continue
    for name in match matched_img; do
      target="${scene_dir}${name}"
      if [[ -d "$target" ]]; then
        rm -rf "$target"
        echo "Removed temp $target"
        removed=$((removed + 1))
      fi
    done
  done
  if [[ "$removed" -eq 0 ]]; then
    echo "No per-scene match temp under $root"
  fi
}

cleanup_on_exit() {
  if [[ -n "${RENDER_ROOT:-}" ]]; then
    cleanup_match_dirs "$RENDER_ROOT"
  fi
}
trap cleanup_on_exit EXIT

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
  echo "CBSR and PD -> $RENDER_ROOT"
  python metrics/eval_consistency.py --dataset "$DATASET" --mode "$MODE" \
    --render-root "$RENDER_ROOT" "${val_args[@]}"
}

echo "Mode: $MODE -> $RENDER_ROOT"
if [[ -n "$CHECKPOINT" ]]; then
  echo "Checkpoint: $CHECKPOINT"
fi
if [[ "$SKIP_INFER" -eq 0 ]]; then
  echo "Inference -> $INFER_SCRIPT --dataset $DATASET"
  launch=(python "$INFER_SCRIPT" --dataset "$DATASET" --output-dir "$RENDER_ROOT")
  if [[ -n "$CHECKPOINT" ]]; then
    launch+=(--checkpoint "$CHECKPOINT")
  fi
  if ((${#INFER_ARGS[@]})); then
    launch+=("${INFER_ARGS[@]}")
  fi
  "${launch[@]}"
fi
run_metrics
