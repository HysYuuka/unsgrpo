#!/usr/bin/env bash
set -euo pipefail
# Keep the private screen PTY open independently of SSH; all visible output is logged.
exec 3<>/dev/tty
export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec /root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/round3_front_driver.py </dev/null >/root/autodl-tmp/unsgrpo-work/runs/round3_rl_method_v1/front_driver.log 2>&1
