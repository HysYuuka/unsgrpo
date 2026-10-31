# Source this file before preparing data or starting an experiment:
#   source /root/autodl-tmp/unsgrpo/storage-env.sh
# Capacity budgets, not enforced quotas:
#   File storage (200 GiB): models 80, datasets 50, archives 50, reserve 20.
#   Local disk (50 GiB): code/envs/caches 15, active runs/checkpoints 25,
#                       temporary files/free space 10.
# Archive completed runs to UNSGRPO_ARCHIVE_DIR after verifying the copy.

export UNSGRPO_PROJECT_DIR=/root/autodl-tmp/unsgrpo
export UNSGRPO_STORAGE_DIR=/root/autodl-fs/unsgrpo
export UNSGRPO_WORK_DIR=/root/autodl-tmp/unsgrpo-work
export UNSGRPO_DATA_DIR="$UNSGRPO_STORAGE_DIR/datasets"
export UNSGRPO_MODEL_DIR="$UNSGRPO_STORAGE_DIR/models"
export UNSGRPO_ARCHIVE_DIR="$UNSGRPO_STORAGE_DIR/archives"
export UNSGRPO_RUN_DIR="$UNSGRPO_WORK_DIR/runs"
export UNSGRPO_CHECKPOINT_DIR="$UNSGRPO_WORK_DIR/checkpoints"
export UNSGRPO_ENV_DIR="$UNSGRPO_WORK_DIR/envs"
export UNSGRPO_PREPARED_DIR="$UNSGRPO_WORK_DIR/prepared"
export UNSGRPO_PSEUDO_DIR="$UNSGRPO_DATA_DIR/pseudo"
export UNSGRPO_EVAL_LABEL_DIR="$UNSGRPO_DATA_DIR/labels_eval_only"
export UNSGRPO_MANIFEST_DIR="$UNSGRPO_PROJECT_DIR/manifests"
export HL_DATA="${HL_DATA:-$UNSGRPO_DATA_DIR/trainer_inputs}"
export HL_OUTPUT="${HL_OUTPUT:-$UNSGRPO_CHECKPOINT_DIR}"

# Explicit budgets in GiB, independent of the root overlay filesystem.
export UNSGRPO_LOCAL_CAPACITY_GIB=50
export UNSGRPO_LOCAL_ENV_CACHE_GIB=15
export UNSGRPO_LOCAL_ACTIVE_GIB=25
export UNSGRPO_LOCAL_RESERVE_GIB=10
export UNSGRPO_FILE_CAPACITY_GIB=200
export UNSGRPO_FILE_MODELS_GIB=80
export UNSGRPO_FILE_DATASETS_GIB=50
export UNSGRPO_FILE_ARCHIVES_GIB=50
export UNSGRPO_FILE_RESERVE_GIB=20

export HF_HOME="$UNSGRPO_WORK_DIR/cache/huggingface"
export HF_HUB_CACHE="$UNSGRPO_MODEL_DIR/hub"
export HF_DATASETS_CACHE="$UNSGRPO_WORK_DIR/cache/datasets"
export TORCH_HOME="$UNSGRPO_WORK_DIR/cache/torch"
export TMPDIR="$UNSGRPO_WORK_DIR/tmp"
