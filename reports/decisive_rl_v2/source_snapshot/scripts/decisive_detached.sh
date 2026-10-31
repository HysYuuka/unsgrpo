#!/usr/bin/env bash
set -eu
exec 3</dev/tty
exec </dev/null
export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HF=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd /root/autodl-tmp/unsgrpo
exec /root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python -u scripts/decisive_await.py > /root/autodl-tmp/unsgrpo-work/runs/decisive_rl_v2/logs/startup.log 2>&1
