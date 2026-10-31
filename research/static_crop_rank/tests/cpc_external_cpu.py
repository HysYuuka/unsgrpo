"""Independent CPC label arithmetic, grouping, metric and execution-gate checks."""
from pathlib import Path
import copy,importlib.util,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('cpc_external',ROOT/'scripts/evaluate_cpc_external.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from croprank.core import metrics

manifest=Path('/root/autodl-fs/highlight-static-rank/data/cpc_external_manifest.jsonl')
records=[json.loads(l) for l in manifest.read_text().splitlines()]
assert len(records)==len({r['source_id'] for r in records})
assert all(r['n']==24 for r in records)
raters=set()
for r in records:
 raters.add(len(r['rater_scores']))
 for i,c in enumerate(r['candidates']):
  assert abs(c['mos']-sum(s[i] for s in r['rater_scores'])/len(r['rater_scores']))<1e-12
  x1,y1,x2,y2=c['bbox'];assert 0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height'];assert all(v==int(v) for v in c['bbox'])
assert {4,5,6,8,9,10}==raters

# Two distinct original records share one source. Each source must receive one
# vote in means/bootstrap, irrespective of image count or three aligned seeds.
fixture=[{'source_id':s,'source_group':g,'overlap_with_GAIC_TEST':s=='d'} for s,g in [('a','g1'),('b','g1'),('c','g2'),('d','g3')]]
y=[4,3.5,3,2,1,0];tables={}
for variant in ('FULL','MATCHED','MUSIQ_FT'):
 for seed in (17,29,43):
  rows=[]
  for r in fixture:
   good=(r['source_id']!='c') if variant=='FULL' else (r['source_id']!='b')
   scores=[0]*6 if variant=='MUSIQ_FT' else ([5,4,3,2,1,0] if good else [0,1,2,3,4,5])
   rows.append({**r,'scores':scores,'mos':y,'metrics':metrics(scores,y)})
  tables[f'{variant}_s{seed}']=rows
result=module.summarize(tables,fixture)
assert abs(result['runs']['FULL_s17']['acc1_5']-2/3)<1e-12
assert abs(result['runs']['MATCHED_s17']['acc1_5']-5/6)<1e-12
comparison=result['cohorts']['all_frozen_external_sources'];assert comparison['source_groups']==3
assert abs(comparison['comparisons']['MATCHED']['acc1_5']['mean_delta']+1/6)<1e-12
assert result['cohorts']['also_excluding_possible_GAIC_TEST_overlap']['source_groups']==2
corrupt=copy.deepcopy(tables);corrupt['FULL_s17'][0]['metrics']['acc1_5']+=.1
try:module.summarize(corrupt,fixture)
except AssertionError:pass
else:raise AssertionError('Independent metric check failed to reject corrupt metrics')
assert json.loads((ROOT/'STATUS.json').read_text())['state']!='CONTROLLED_MATRIX_COMPLETE'
try:module.main()
except AssertionError as error:assert 'before the core matrix' in str(error)
else:raise AssertionError('Premature external inference was not blocked')
print(json.dumps({'status':'PASSED_CPU_ONLY','originals_independently_rechecked':len(records),'native_candidate_scores_checked':sum(r['n'] for r in records),
 'source_group_weighting_fixture':True,'aligned_seeds_not_added_as_samples':True,'independent_metrics_reject_corruption':True,'early_inference_gate_blocks_before_GPU_import':True,'GPU_work':False},indent=2))
