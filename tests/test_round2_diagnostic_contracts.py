import importlib.util
from pathlib import Path
import numpy as np
import pytest
from unsgrpo.core import midrank_percentile
from unsgrpo.diagnostics import StableMapper,aggregate,offset_consensus,student_loss
from unsgrpo.round2_utils import prototypes,teacher_scores,overlap_statistics
from unsgrpo.evaluation import average_precision_noninterpolated


def registered_mapper():
    p=Path('/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1/evaluation_sensitivity_v01/evaluate_sensitivity.py')
    if not p.exists():pytest.skip('Original registered local artifact not available')
    spec=importlib.util.spec_from_file_location('registered_mapper_for_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.stable_interval_average


def test_cached_mapper_matches_registered_on_synthetic_and_real_clocks():
    fn=registered_mapper();rng=np.random.default_rng(20261028)
    for _ in range(10):
        s=np.r_[0,np.cumsum(rng.uniform(.2,2,20))];t=np.r_[0,np.sort(rng.uniform(0,s[-1],100)),s[-1]]
        src=np.c_[s[:-1],s[1:]];dst=np.c_[t[:-1],t[1:]];q=rng.uniform(0,1,20)
        assert StableMapper(src,dst)(q)==pytest.approx(fn(src,q,dst),abs=2e-15)
        assert np.array_equal(StableMapper(src,dst)([.1234567]*20),np.full(len(dst),.1234567))
    import json
    manifest=Path('/root/autodl-tmp/unsgrpo-work/prepared/prepared.jsonl')
    if not manifest.exists():return
    for row in map(json.loads,manifest.read_text().splitlines()):
        if row['split']!='test':continue
        dst=np.load(row['frame_bounds_path'],allow_pickle=False);q=rng.uniform(0,1,len(row['bounds']))
        assert StableMapper(row['bounds'],dst)(q)==pytest.approx(fn(row['bounds'],q,dst),abs=2e-15)


def test_prototype_length_and_self_exclusion_are_different_factors():
    f={'long':np.array([[1.,0.]]*3),'short':np.array([[0.,1.]]),'peer':np.array([[1.,1.]])/np.sqrt(2)}
    c={k:0 for k in f};ids=list(f)
    clip,_=prototypes(f,ids,c);equal,_=prototypes(f,ids,c,video_equal=True)
    expected=(3*f['long'][0]+f['short'][0]+f['peer'][0])/5
    assert clip[0]==pytest.approx(expected)
    assert equal[0]==pytest.approx(np.mean([x.mean(0) for x in f.values()],0))
    exc,_=prototypes(f,ids,c,exclude='long');veexc,_=prototypes(f,ids,c,exclude='long',video_equal=True)
    assert exc[0]==pytest.approx((f['short'][0]+f['peer'][0])/2)
    assert exc[0]==pytest.approx(veexc[0])
    assert not np.allclose(clip[0],equal[0])


def test_teacher_scores_invariant_to_cluster_label_permutation():
    f={'a':np.array([[1.,0.],[0.,1.]]),'b':np.array([[.6,.8],[.8,.6]]),'c':np.array([[1.,0.],[.8,.6]])}
    c={'a':0,'b':0,'c':1};d={'a':4,'b':4,'c':9}
    p,_=prototypes(f,list(f),c);q,_=prototypes(f,list(f),d)
    a=teacher_scores(f,list(f),c,p);b=teacher_scores(f,list(f),d,q)
    for k in f:assert np.array_equal(a[k]['q'],b[k]['q'])


def test_affine_scale_reduces_raw_spread_but_not_rank_inconsistency():
    ids=[[0,1,2],[1,2,3]];values=[np.array([.1,.6,.4]),np.array([.3,.5,.9])]
    a=overlap_statistics(ids,values);b=overlap_statistics(ids,[.25*v+.375 for v in values])
    assert b['mean_range']==pytest.approx(.25*a['mean_range'])
    assert b['scale_normalized_mean_range']==pytest.approx(a['scale_normalized_mean_range'])
    assert b['overlap_pair_order_disagreement']==a['overlap_pair_order_disagreement']


def test_overlap_correction_affine_equivariance_and_zero_gauge():
    ids=[[0,1,2],[1,2,3]];values=[np.array([.3,.4,.5]),np.array([0.,.1,.2])]
    a,offset=offset_consensus(ids,values);b,bo=offset_consensus(ids,[.25*v+.375 for v in values])
    assert offset.sum()==pytest.approx(0,abs=1e-15)
    assert bo==pytest.approx(.25*offset)
    assert aggregate(4,ids,b)==pytest.approx(.25*aggregate(4,ids,a)+.375)


def test_monotone_ap_invariant_and_quantization_is_not_strictly_monotone():
    s=np.array([.495,.5,.8,.9]);y=np.array([0,1,0,1])
    assert average_precision_noninterpolated(y,s)==average_precision_noninterpolated(y,.25*s+.375)
    t=np.round(.25*s+.375,2)
    assert t[0]==t[1] and s[0]!=s[1]


@pytest.mark.parametrize('function',[aggregate,offset_consensus])
def test_no_silent_zip_drop_or_nonfinite_offset_observations(function):
    args=(3,[[0,1]],[]) if function==aggregate else ([[0,1]],[])
    with pytest.raises(ValueError):function(*args)
    args=(3,[[0,1],[1,2]],[[0,np.nan],[0,1]]) if function==aggregate else ([[0,1]],[[0,np.nan]])
    with pytest.raises(ValueError):function(*args)


def test_pure_ranking_gradients_and_tied_target_finite():
    import torch
    pred=torch.tensor([[.8,.2]],requires_grad=True);q=torch.tensor([[.2,.8]])
    student_loss(pred,q,q,'rank_pair').backward()
    assert pred.grad[0,0]>0 and pred.grad[0,1]<0
    pred=torch.tensor([[.8,.2]],requires_grad=True);q=torch.full_like(pred,.5)
    loss=student_loss(pred,q,q,'rank_pair');loss.backward()
    assert loss==0 and torch.equal(pred.grad,torch.zeros_like(pred))


def test_student_capacity_all_projected_dimensions_active():
    import torch
    path=Path(__file__).resolve().parents[1]/'scripts/round2_train_students.py'
    spec=importlib.util.spec_from_file_location('small_student_for_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(42);net=m.SmallStudent()
    assert sum(p.numel() for p in net.parameters())==32897
    x=torch.randn(16,8,512);net(x).sum().backward()
    assert torch.all(net.net[0].weight.grad.abs().sum(0)>0)
