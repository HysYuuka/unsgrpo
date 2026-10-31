#!/usr/bin/env bash
set -euo pipefail
exec 3<>/dev/tty
export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec /root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/reward_to_rl_driver.py </dev/null >/root/autodl-tmp/unsgrpo-work/runs/reward_to_rl_v1/driver.log 2>&1
