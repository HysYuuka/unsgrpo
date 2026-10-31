from pathlib import Path
import sys,json,time,math,gc
import numpy as np
from PIL import Image,ImageDraw
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.model import CropRank,sample_field,grid_xy,boundary_points,coverage
from croprank.core import seed_all,losses,metrics
from croprank.replay import replay_backward,predict
ROOT=Path(__file__).resolve().parents[1];report={'status':'running','checks':{}}
def record(k,v):
 report['checks'][k]=v;(ROOT/'reports/ACCEPTANCE_NUMERIC.json').write_text(json.dumps(report,indent=2));print(k,{a:b for a,b in v.items() if a!='per_parameter'},flush=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
seed_all(17)
# Independent physical-coordinate fixtures.
y,x=torch.meshgrid(torch.arange(10),torch.arange(12),indexing='ij');field=(3*x+2*y).float()[None,None]
meta={'scale':(2.,3.),'size':(100,100)};xy=torch.tensor([[[4.25,2.833333333],[8.25,5.5]]])
z=sample_field(field,xy,meta,4,.5).flatten();expected=torch.tensor([10.,20.]);assert torch.allclose(z,expected,atol=1e-6),(z,expected)
b=torch.tensor([[20.,10.,100.,80.],[0.,0.,128.,96.]])
bp,anchor=boundary_points(b)
assert (bp[:,0,0,1]>anchor[:,0,1]).all() and (bp[:,16,0,0]<anchor[:,16,0]).all() and (bp[:,32,0,1]<anchor[:,32,1]).all() and (bp[:,48,0,0]>anchor[:,48,0]).all()
a,area=coverage(b,{'size':(128,96),'scale':(1,1)},6,8);assert torch.equal(a[1],torch.ones_like(a[1]));assert 0<a[0].mean()<1
record('coordinates',{'ramp_max_error':float((z-expected).abs().max()),'four_edge_orientations':True,'full_frame_coverage':True})
# Hand-calculated metric fixture including equal-score and MOS ties.
s=[9,8,7,6,5,4,3,2,1,0,0,0];mos=[1,2,3,4,5,5,4,3,2,1,1,1];m=metrics(s,mos)
assert m['acc1_5']==0 and m['acc1_10']==1 and m['regret']==4
assert metrics([1]*12,mos)['srcc']==0
score=torch.tensor([-.1,.2,.4],requires_grad=True);_,parts=losses(score,torch.tensor([1.,3.,5.]));g=torch.autograd.grad(parts['sel'],score)[0];assert g.norm()>0
record('metrics_and_selection',{'hand_fixture':m,'selection_gradient':g.tolist()})
# A real source image, full two resolutions. Four candidates are ONLY a graph equivalence fixture.
from PIL import Image
p=Path('/root/autodl-fs/highlight-aesthetic-rl/assets/gaic/GAIC/images/train/210429.jpg');im=Image.open(p).convert('RGB');w,h=im.size
boxes=torch.tensor([[.08*w,.06*h,.91*w,.91*h],[0,0,w,h],[.1*w,.12*h,.8*w,.85*h],[.15*w,.15*h,.95*w,.95*h]],device='cuda');mos=torch.tensor([2.,4.,3.,5.],device='cuda')
overlay=im.copy();d=ImageDraw.Draw(overlay)
for box in boxes.cpu().tolist():d.rectangle(box,outline='red',width=3)
overlay.save(ROOT/'reports/acceptance_overlay.jpg')
model=CropRank().cuda().train();record('parameters',{'FULL':sum(p.numel() for p in model.parameters()),'all_trainable':all(p.requires_grad for p in model.parameters())})
model.zero_grad(set_to_none=True);out=model(im,boxes);loss,_=losses(out,mos);loss.backward()
ref={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None};missing=[n for n,p in model.named_parameters() if p.grad is None];assert not missing,missing
ref_s=out.detach().cpu();ref_l=float(loss.detach());del out,loss;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
terms,scores=replay_backward(model,im,boxes,mos,chunk=2,amp=False)
errors={};total_num=total_den=0.
for n,p in model.named_parameters():
 assert p.grad is not None and torch.isfinite(p.grad).all(),n
 g=p.grad.detach().cpu();r=ref[n];num=float((g-r).double().square().sum());den=float(r.double().square().sum());rel=(num/max(den,1e-30))**.5
 errors[n]={'relative_l2':rel,'max_abs':float((g-r).abs().max())};total_num+=num;total_den+=den
# Tiny/zero gradients are also logged; global threshold plus each material tensor.
worst=max(v['relative_l2'] for n,v in errors.items() if ref[n].norm()>1e-7)
record('gradient_replay_fp32',{'relative_l2':(total_num/total_den)**.5,'worst_material_tensor':worst,'score_max_abs':float((scores.cpu()-ref_s).abs().max()),'loss_error':abs(terms['total']-ref_l),'per_parameter':errors})
assert (total_num/total_den)**.5<=1e-4 and worst<=1e-4
# All declared modules have an optimizer update; no freeze in formal training.
before={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('weight') and p.grad.abs().max()>0}
opt=torch.optim.AdamW(model.parameters(),lr=1e-4);opt.step();changed=sum(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before);assert changed==len(before)
record('gradient_coverage',{'parameters_with_grad':len(ref),'nonzero_weight_updates':changed,'expected':len(before)})
del before,opt,ref;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
model.eval()
with torch.no_grad():
 f,meta=model.encode(im);s1,debug=model.score_fields(f,meta,boxes,debug=True)
 assert meta['semantic']['long_edge']==512 and meta['detail']['long_edge']==1024
 assert debug['empty'][1,1] and debug['boundary'].shape==(4,64,256)
 assert not torch.allclose(debug['source'][0]-debug['source'][2],(debug['source'][0]-debug['source'][2]).mean(0,keepdim=True).expand_as(debug['source'][0]))
 errors={str(c):float((predict(model,im,boxes,c,amp=False)-s1).abs().max()) for c in [1,2,4]}
 perm=torch.tensor([2,0,3,1],device='cuda');permuted=model.score_fields(f,meta,boxes[perm]);duplicate=model.score_fields(f,meta,torch.cat([boxes,boxes[:1]]))
 assert max(errors.values())<1e-5 and torch.allclose(permuted,s1[perm],atol=1e-5) and torch.allclose(duplicate[:4],s1,atol=1e-5)
 record('candidate_invariance',{'chunk_error':errors,'permutation_error':float((permuted-s1[perm]).abs().max()),'duplicate_error':float((duplicate[:4]-s1).abs().max()),'detail_shape':list(f[1].shape),'layout_shape':list(debug['layout'].shape),'empty_full_frame_discard':True,'candidate_source_nonconstant':True})
 del f,debug
model.train();terms,s=replay_backward(model,im,boxes,mos,chunk=2,amp=True);assert np.isfinite(terms['total'])
record('bf16',{'score_max_error_from_fp32':float((s-s1).abs().max()),'loss':terms,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
report['status']='passed';(ROOT/'reports/ACCEPTANCE_NUMERIC.json').write_text(json.dumps(report,indent=2));print('ACCEPTANCE PASS',flush=True)
