#!/usr/bin/env bash
# Stage commands for ms-swift 4.6.0. Formal runs require hashed preflight/freeze evidence.
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNSGRPO_CALLER_HL_DATA_SET=${HL_DATA+x}
UNSGRPO_CALLER_HL_OUTPUT_SET=${HL_OUTPUT+x}
UNSGRPO_CALLER_HL_DATA=${HL_DATA-}
UNSGRPO_CALLER_HL_OUTPUT=${HL_OUTPUT-}
source "$PROJECT/storage-env.sh"
if [[ $UNSGRPO_CALLER_HL_DATA_SET == x ]]; then export HL_DATA="$UNSGRPO_CALLER_HL_DATA"; fi
if [[ $UNSGRPO_CALLER_HL_OUTPUT_SET == x ]]; then export HL_OUTPUT="$UNSGRPO_CALLER_HL_OUTPUT"; fi
export PATH="$UNSGRPO_WORK_DIR/envs/highlights/bin:$PATH"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
STAGE=${1:?stage: smoke-sft0|smoke-grpo|smoke-merge|sft0|merge|sft-continue|grpo}
MODEL=${2:?local model directory required}
ADAPTER=${3:-}
MODE=formal
case "$STAGE" in
  smoke-sft0) MODE=smoke; ACTION=sft0 ;;
  smoke-grpo) MODE=smoke; ACTION=grpo ;;
  smoke-merge) MODE=smoke; ACTION=merge ;;
  sft0|merge|sft-continue|grpo) ACTION=$STAGE ;;
  *) echo "Unsupported stage: $STAGE" >&2; exit 2 ;;
esac
SEED=${HL_SEED:-42}
G=${HL_GROUP_SIZE:-4}
PROMPTS=${HL_PROMPTS:-4}
STEPS=${HL_STEPS:-256}
SFT_BATCH=${HL_SFT_BATCH:-16}
if [[ $MODE == smoke ]]; then
  PROMPTS=${HL_PROMPTS:-1}
  STEPS=${HL_STEPS:-4}
  SFT_BATCH=${HL_SFT_BATCH:-1}
fi
WORLD=${NPROC_PER_NODE:-1}
[[ $WORLD == 1 ]] || { echo 'This first-round driver is audited for one GPU only.' >&2; exit 2; }
DATA=${HL_DATA:-$UNSGRPO_WORK_DIR/model_inputs}
OUT=${HL_OUTPUT:-$UNSGRPO_CHECKPOINT_DIR}
AUDIT=${HL_INPUT_AUDIT:-$PROJECT/reports/input_audit.json}
FREEZE=${HL_FREEZE:-$PROJECT/reports/freeze.json}
RANK_WEIGHT=${HL_RANK_WEIGHT:-0.8}
export HL_GROUP_SIZE=$G HL_RANK_WEIGHT=$RANK_WEIGHT
export HL_GRPO_DATA="$DATA/train_grpo.jsonl"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-784}
export VIDEO_MIN_TOKEN_NUM=${VIDEO_MIN_TOKEN_NUM:-64}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-64}
export FPS=2 FPS_MAX_FRAMES=32 FPS_MIN_FRAMES=4
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# No run directory/receipt is created until the gates pass. Runtime failures are
# then recorded as FAILED and never mistaken for completed optimizer updates.
python - "$PROJECT" "$MODE" "$ACTION" "$MODEL" "$DATA" "$OUT" "$AUDIT" "$FREEZE" "$STEPS" "$PROMPTS" "$SFT_BATCH" "$G" "$SEED" "$RANK_WEIGHT" <<'PY'
import hashlib,json,sys
from pathlib import Path
project,mode,stage,model,data,out,audit,freeze,steps,prompts,batch,g,seed,weight=sys.argv[1:]
project,model,data,out=map(lambda x:Path(x).resolve(),(project,model,data,out))
if not model.is_dir(): raise SystemExit('Model directory does not exist; remote main is not allowed')
if out==project or project in out.parents: raise SystemExit('HL_OUTPUT must be outside the code repository')
if any(int(v)<1 for v in (steps,prompts,batch,g)): raise SystemExit('Invalid positive budget')
if int(g)!=4: raise SystemExit('First-round protocol requires G=4')
if not Path(audit).is_file(): raise SystemExit('Missing real full-input processor/template audit')
a=json.loads(Path(audit).read_text())
if a.get('status')!='COMPLETED' or not a.get('all_rows_encoded') or a.get('errors'):
 raise SystemExit('Full-input audit did not pass')
import os
visual_keys=('IMAGE_MAX_TOKEN_NUM','VIDEO_MIN_TOKEN_NUM','VIDEO_MAX_TOKEN_NUM','FPS','FPS_MAX_FRAMES','FPS_MIN_FRAMES')
if any(a.get('visual_environment',{}).get(k)!=os.environ.get(k) for k in visual_keys):
 raise SystemExit('Visual environment differs from the actual input audit')
rows_path=Path(os.environ.get('HL_INPUT_AUDIT_ROWS',a.get('rows_path',''))).resolve()
if not rows_path.is_file() or hashlib.sha256(rows_path.read_bytes()).hexdigest()!=a.get('rows_sha256'):
 raise SystemExit('Actual input-audit row ledger is missing or its hash changed')
needed='train_grpo.jsonl' if stage=='grpo' else 'train_sft.jsonl'
if stage!='merge':
 path=(data/needed).resolve()
 if not path.is_file(): raise SystemExit('Missing registered training data')
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if a.get('input_sha256',{}).get(str(path))!=digest: raise SystemExit('Training file differs from audited file')
if mode=='formal':
 f=json.loads(Path(freeze).read_text())
 if f.get('status')!='FROZEN': raise SystemExit('Protocol is not FROZEN')
 if not f.get('files'): raise SystemExit('Freeze has no immutable file hashes')
 frozen_paths={str((Path(p) if Path(p).is_absolute() else project/p).resolve()) for p in f['files']}
 required_frozen=[Path(audit).resolve(),rows_path,project/'scripts/run.sh',project/'scripts/reward_plugin.py',project/'scripts/audit_swift.py',project/'src/unsgrpo/core.py',project/'src/unsgrpo/data.py',project/'scripts/training_telemetry.py',project/'scripts/restore_processor.py',project/'scripts/swift_compat.py']
 if stage!='merge': required_frozen.append((data/needed).resolve())
 if any(str(p) not in frozen_paths for p in required_frozen): raise SystemExit('Freeze must hash the audited dataset, actual input audit and executed training/reward/core code')
 for path,digest in f['files'].items():
  p=Path(path);p=p if p.is_absolute() else project/p
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
   raise SystemExit(f'Frozen file changed: {path}')
 receipt=Path(f['technical_acceptance'])
 if not receipt.is_absolute(): receipt=project/receipt
 if str(receipt.resolve()) not in frozen_paths: raise SystemExit('Freeze must hash the real GPU acceptance receipt')
 t=json.loads(receipt.read_text())
 required=('generation_reward_logps_backward_optimizer_save_reload','same_group_media_q_id',
           'assistant_only_sft_no_grpo_answers','strict_json_equal_reward_truncation',
           'finite_lora_gradients_parameter_update_frozen_backbone','s0_bf16_tensor_merge_common_endpoint',
           'fixed_s0_reference_after_update','old_ref_reward_advantage_detached',
           'completion_only_loss_eos_lengths_std_sampling','stage_memory_time_recorded',
           'formal_rollout_concurrency_verified')
 if t.get('status')!='COMPLETED' or t.get('gpu_training_verified') is not True:
  raise SystemExit('Real GPU technical loop has not passed')
 if any(t.get('checks',{}).get(k) is not True for k in required):
  raise SystemExit('Required GPU acceptance evidence is incomplete')
 # Budget is frozen per branch. Neither tests nor smoke settings can silently
 # replace the original 256-update comparison after the freeze.
 key='G-CAL' if stage=='grpo' and float(weight)==0 else 'G-RANK' if stage=='grpo' else {'sft0':'S0','sft-continue':'S+'}.get(stage)
 if key:
  spec=f['matrix'][key]
  actual={'updates':int(steps),'prompt_exposures_per_update':int(batch if stage=='sft0' else prompts),
          'group_size':int(g) if stage=='grpo' else 1,'seed':int(seed)}
  if stage=='grpo': actual['rank_weight']=float(weight)
  if any(spec.get(k)!=v for k,v in actual.items()): raise SystemExit(f'Runtime budget differs from frozen {key}: {actual}')
 if stage in {'sft-continue','grpo'}:
  identity=json.loads((project/'reports/common_s0.json').read_text())
  if identity.get('status')!='COMPLETED' or Path(identity['merged_model']).resolve()!=model or identity.get('merge_validation_passed') is not True or identity.get('acceptance_basis')!='verified_bf16_tensor_merge_common_s0':
   raise SystemExit('Continuation model is not the registered common merged S0')
print(json.dumps({'gate':'PASSED','mode':mode,'stage':stage,'model':str(model)},ensure_ascii=False))
PY
RUN_ID=${HL_RUN_ID:-${MODE}_${ACTION}_rank${RANK_WEIGHT}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$UNSGRPO_RUN_DIR/$RUN_ID"
[[ ! -e $RUN_DIR ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR" "$OUT"
export HL_ROLLOUT_LEDGER="$RUN_DIR/rollouts.jsonl"
export HL_COMPAT_RECEIPT="$RUN_DIR/swift_compat.json"
export HL_INPUT_AUDIT_ROWS=${HL_INPUT_AUDIT_ROWS:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["rows_path"])' "$AUDIT")}
export HL_TRAINING_LEDGER="$RUN_DIR/training_ledger.jsonl"
export HL_TELEMETRY_SUMMARY="$RUN_DIR/training_summary.json"
SAVE_STEPS=128
[[ $MODE != smoke ]] || SAVE_STEPS=$STEPS
OUTPUT_DIR="$OUT/$RUN_ID"
common=(--model "$MODEL" --tuner_type lora --target_modules all-linear
        --freeze_vit true --freeze_aligner true --freeze_llm false
        --lora_rank 16 --lora_alpha 32 --lora_dropout 0
        --bf16 true --torch_dtype bfloat16 --attn_impl sdpa --gradient_checkpointing true
        --per_device_train_batch_size 1 --max_length 8192
        --packing false --padding_free false --split_dataset_ratio 0
        --dataset_num_proc 1 --dataloader_num_workers 2 --seed "$SEED" --data_seed 42
        --max_steps "$STEPS" --logging_steps 1 --save_steps "$SAVE_STEPS" --save_total_limit 2
        --callbacks highlight_telemetry
        --eval_strategy no --lr_scheduler_type cosine --warmup_ratio 0.05
        --weight_decay 0 --max_grad_norm 1 --strict true --report_to tensorboard)
case "$ACTION" in
  sft0)
    CMD=(swift sft "${common[@]}" --dataset "$DATA/train_sft.jsonl"
         --external_plugins "$PROJECT/scripts/swift_compat.py" "$PROJECT/scripts/training_telemetry.py"
         --learning_rate 5e-5 --gradient_accumulation_steps "$SFT_BATCH" --output_dir "$OUTPUT_DIR") ;;
  merge)
    [[ -n $ADAPTER ]] || { echo 'merge requires an actual adapter path as the third argument' >&2; exit 2; }
    CMD=(swift export --model "$MODEL" --adapters "$ADAPTER" --merge_lora true --output_dir "$OUTPUT_DIR") ;;
  sft-continue)
    CMD=(swift sft "${common[@]}" --dataset "$DATA/train_sft.jsonl"
         --external_plugins "$PROJECT/scripts/swift_compat.py" "$PROJECT/scripts/training_telemetry.py"
         --learning_rate 1e-5 --gradient_accumulation_steps "$PROMPTS" --output_dir "$OUTPUT_DIR") ;;
  grpo)
    COMPLETIONS=$((PROMPTS*G))
    CMD=(swift rlhf "${common[@]}" --rlhf_type grpo --dataset "$HL_GRPO_DATA"
         --external_plugins "$PROJECT/scripts/swift_compat.py" "$PROJECT/scripts/reward_plugin.py" "$PROJECT/scripts/training_telemetry.py" --reward_funcs highlight_recurrence
         --remove_unused_columns false --learning_rate 1e-5
         --gradient_accumulation_steps "$COMPLETIONS" --generation_batch_size "$COMPLETIONS"
         --num_generations "$G" --use_vllm false --loss_type grpo --advantage_estimator grpo
         --scale_rewards group --kl_in_reward false --importance_sampling_level token
         --num_iterations 1 --beta 0.04 --epsilon 0.2 --dynamic_sample false
         --overlong_filter false --temperature 1 --top_p 1 --top_k 0
         --repetition_penalty 1 --max_completion_length 128 --truncation_strategy left
         --output_dir "$OUTPUT_DIR") ;;
esac
# Version checked at execution, without installing or resolving new dependencies.
python - <<'PY'
from importlib.metadata import version
assert version('ms-swift')=='4.6.0', 'Requires audited ms-swift==4.6.0'
PY
export UNSGRPO_RUN_DIR_ACTIVE="$RUN_DIR" UNSGRPO_OUTPUT_ACTIVE="$OUTPUT_DIR" UNSGRPO_RUN_MODE="$MODE" UNSGRPO_RUN_STAGE="$ACTION"
export UNSGRPO_PROCESSOR_RESTORE_SCRIPT="$PROJECT/scripts/restore_processor.py"
export UNSGRPO_INPUT_AUDIT_ACTIVE="$AUDIT" UNSGRPO_TRAIN_SFT_ACTIVE="$DATA/train_sft.jsonl"
export UNSGRPO_FREEZE_ACTIVE="$FREEZE"
python - "${CMD[@]}" <<'PY'
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['UNSGRPO_RUN_DIR_ACTIVE'])
receipt={'status':'RUNNING','mode':os.environ['UNSGRPO_RUN_MODE'],'stage':os.environ['UNSGRPO_RUN_STAGE'],
         'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv[1:],
         'output_dir':os.environ['UNSGRPO_OUTPUT_ACTIVE'],'gpu_training_verified':False,
         'n_successful_optimizer_updates':None,'n_prompt_exposures':None,
         'note':'Trainer exit success alone does not verify exposure or GPU acceptance; audit optimizer/update ledgers.',
         'env':{k:os.environ[k] for k in ('HL_GRPO_DATA','HL_GROUP_SIZE','HL_RANK_WEIGHT','HL_ROLLOUT_LEDGER',
                 'IMAGE_MAX_TOKEN_NUM','VIDEO_MIN_TOKEN_NUM','VIDEO_MAX_TOKEN_NUM','FPS','FPS_MAX_FRAMES','FPS_MIN_FRAMES')}}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as source:
  for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
receipt.update(run_id=root.name,exact_resume_verified=False,input_audit_path=str(Path(os.environ['UNSGRPO_INPUT_AUDIT_ACTIVE']).resolve()),
               input_audit_sha256=digest(os.environ['UNSGRPO_INPUT_AUDIT_ACTIVE']))
if receipt['stage'] in {'sft0','sft-continue','grpo'}:
 training_data=Path(sys.argv[sys.argv.index('--dataset')+1]).resolve()
 receipt.update(training_dataset_path=str(training_data),training_dataset_sha256=digest(training_data))
if receipt['mode']=='formal':
 frozen=Path(os.environ['UNSGRPO_FREEZE_ACTIVE']).resolve()
 receipt.update(freeze_path=str(frozen),freeze_sha256=digest(frozen))
 receipt['method_id']=({'sft0':'S0','sft-continue':'S+'}.get(receipt['stage']) or
                       ('G-CAL' if float(os.environ['HL_RANK_WEIGHT'])==0 else 'G-RANK') if receipt['stage']!='merge' else None)
 if receipt['stage'] in {'sft-continue','grpo'}:
  common=Path(os.environ['UNSGRPO_PROJECT_DIR'])/'reports/common_s0.json'
  receipt.update(parent_common_s0_path=str(common.resolve()),parent_common_s0_sha256=digest(common))
def save(): (root/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
save();started=time.monotonic()
try:
 with (root/'console.log').open('w') as log:
  proc=subprocess.Popen(sys.argv[1:],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in proc.stdout:
   sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
  code=proc.wait()
  if code==0 and receipt['stage']=='merge':
   command=[sys.executable,os.environ['UNSGRPO_PROCESSOR_RESTORE_SCRIPT'],'--exported',receipt['output_dir'],
            '--input-audit',os.environ['UNSGRPO_INPUT_AUDIT_ACTIVE'],'--inputs',os.environ['UNSGRPO_TRAIN_SFT_ACTIVE'],
            '--out',str(root/'processor_restore.json')]
   receipt['processor_restore_command']=command;save()
   restore=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   for line in restore.stdout:
    sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
   code=restore.wait()
   receipt['processor_restore_exit_code']=code
 receipt.update(status='TRAINER_EXITED_SUCCESS_REQUIRES_AUDIT' if code==0 else 'FAILED',exit_code=code)
 receipt['trainer_exit_status']=receipt['status']
except BaseException as exc:
 receipt.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');save();raise
finally:
 receipt['wall_seconds']=time.monotonic()-started;receipt['ended_utc']=datetime.now(timezone.utc).isoformat();save()
sys.exit(code)
PY
