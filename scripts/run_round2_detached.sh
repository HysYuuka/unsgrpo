#!/usr/bin/env bash
set -eu
# screen keeps its own private PTY; the job's standard streams are local files.
exec 3<>/dev/tty
export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HF=1
/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/run_round2_diagnostics.py --execute </dev/null >>/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1/pipeline.log 2>&1
/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/archive_round2_diagnostics.py </dev/null >>/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1/archive.log 2>&1
