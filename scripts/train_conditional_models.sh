#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-embedding}"
METADATA_PATH="${2:-data/nsynth_waveform_box/metadata/metadata.jsonl}"
TRACKING="${3:-wandb}"
EXTRA_ARGS=("${@:4}")
export PYTHONPATH=.

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Error: metadata file not found at: $METADATA_PATH"
  echo "Tip: pass an explicit path as the second argument."
  exit 1
fi

BASE_CMD=(python train.py datamodule.metadata_path="$METADATA_PATH")

resolve_exp_name() {
  local onehot_exp embedding_exp
  case "$TRACKING" in
    wandb)
      onehot_exp="nsynth_conditional_16gb_wandb"
      embedding_exp="nsynth_conditional_16gb_embedding_wandb"
      ;;
    no_wandb)
      onehot_exp="nsynth_conditional_16gb_no_wandb"
      embedding_exp="nsynth_conditional_16gb_embedding_no_wandb"
      ;;
    *)
      echo "Error: unknown tracking mode '$TRACKING'. Expected one of: wandb, no_wandb"
      exit 1
      ;;
  esac

  if [[ "$1" == "onehot" ]]; then
    echo "$onehot_exp"
  else
    echo "$embedding_exp"
  fi
}

run_conditional_onehot() {
  echo "Starting conditional diffusion training (onehot)..."
  local exp_name
  exp_name="$(resolve_exp_name onehot)"
  "${BASE_CMD[@]}" exp="$exp_name" model.conditioning_mode=onehot model.use_contrastive_loss=false "${EXTRA_ARGS[@]}"
}

run_embedding_contrastive() {
  echo "Starting embedding-conditioned + contrastive training..."
  local exp_name
  exp_name="$(resolve_exp_name embedding)"
  "${BASE_CMD[@]}" exp="$exp_name" model.conditioning_mode=label_embedding model.use_contrastive_loss=true "${EXTRA_ARGS[@]}"
}

case "$MODE" in
  onehot)
    run_conditional_onehot
    ;;
  embedding)
    run_embedding_contrastive
    ;;
  both)
    run_conditional_onehot
    run_embedding_contrastive
    ;;
  *)
    echo "Usage: scripts/train_conditional_models.sh [onehot|embedding|both] [metadata_path] [wandb|no_wandb] [extra hydra overrides...]"
    echo "Example: scripts/train_conditional_models.sh both data/nsynth_waveform_processed/metadata/metadata.jsonl wandb +trainer.fast_dev_run=1"
    exit 1
    ;;
esac