#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-embedding}"
METADATA_PATH="${2:-data/nsynth_waveform_box/metadata/metadata.jsonl}"
EXTRA_ARGS=("${@:3}")
export PYTHONPATH=.

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Error: metadata file not found at: $METADATA_PATH"
  echo "Tip: pass an explicit path as the second argument."
  exit 1
fi

BASE_CMD=(python train.py datamodule.metadata_path="$METADATA_PATH")

run_conditional_onehot() {
  echo "Starting conditional diffusion training (onehot)..."
  "${BASE_CMD[@]}" exp=nsynth_conditional_16gb_no_wandb model.conditioning_mode=onehot model.use_contrastive_loss=false "${EXTRA_ARGS[@]}"
}

run_embedding_contrastive() {
  echo "Starting embedding-conditioned + contrastive training..."
  "${BASE_CMD[@]}" exp=nsynth_conditional_16gb_embedding_no_wandb model.conditioning_mode=label_embedding model.use_contrastive_loss=true "${EXTRA_ARGS[@]}"
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
    echo "Usage: scripts/train_conditional_models.sh [onehot|embedding|both] [metadata_path] [extra hydra overrides...]"
    echo "Example: scripts/train_conditional_models.sh both data/nsynth_waveform_processed/metadata/metadata.jsonl +trainer.fast_dev_run=1"
    exit 1
    ;;
esac