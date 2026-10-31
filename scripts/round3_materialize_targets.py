#!/usr/bin/env python3
"""Create exactly one predeclared q/c version from already sealed semantic calls."""
import argparse
import json
import numpy as np
from unsgrpo.round3_common import *
from unsgrpo.round3_teacher import combine


def main(alpha):
    out=STORE/'teacher';sealed=read(out/'semantic.freeze.json')
    for key in ('semantic_call_file','semantic_clip_file','reference_targets'):
        r=sealed[key]
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed sealed semantic input')
    if alpha==.5:
        gate=read(REPORT/'teacher_gate_alpha075.json')
        if gate['passed']:raise ValueError('Do not evaluate fallback if primary passed')
    tag='alpha075' if alpha==.75 else 'alpha050';rr=records(out/'semantic_clips.jsonl');data={};detail=[]
    for k in sorted({r['video_id'] for r in rr}):
        group=[r for r in rr if r['video_id']==k];group.sort(key=lambda r:r['clip_id'])
        qq=[];cc=[]
        for r in group:
            def view(which):
                if r['score_'+which] is None:return None
                return {'scores':[r['score_'+which]],'roles':[r['role_'+which]]}
            q,c,z,u,reason=combine([r['q_excl']],view('a'),view('b'),alpha)
            qq.append(q[0]);cc.append(c[0]);detail.append(dict(r,q1=float(q[0]),c1=float(c[0]),
                semantic_disagreement=abs(r['score_a']-r['score_b']) if r['score_a'] is not None and r['score_b'] is not None else None,
                z=float(z[0]),u=float(u[0]),alpha=alpha,fallback_reason=reason[0]))
        data[f'q__{k}']=np.array(qq);data[f'c__{k}']=np.array(cc)
    dest=out/f'targets_{tag}.npz'
    if dest.exists():raise FileExistsError(dest)
    np.savez_compressed(dest,**data)
    with (out/f'clips_{tag}.jsonl').open('x') as f:
        for r in detail:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write(out/f'targets_{tag}.freeze.json',{'status':'FROZEN_BEFORE_THIS_VERSION_QUALITY_EVALUATION',
        'time':timestamp(),'alpha':alpha,'n_clips':len(rr),'arrays':reference(dest),
        'records':reference(out/f'clips_{tag}.jsonl'),'semantic_freeze':reference(out/'semantic.freeze.json'),
        'teacher_calls_added':0,'human_labels_read':False})
    print(f'{tag}: {len(rr)} frozen clip targets',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--alpha',type=float,required=True,choices=[.75,.5]);a=p.parse_args();main(a.alpha)
