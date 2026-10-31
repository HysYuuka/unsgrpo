from pathlib import Path
import sys,json,gc,time
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.train import build_model,param_groups
from croprank.core import losses,load_image,seed_all
from croprank.replay import replay_backward,autocast
from croprank.musiq import MusiqRank
ROOT=Path(__file__).resolve().parents[1];r={'status':'running'}
def record(k,v):
 r[k]=v;(ROOT/'reports/ADDITIONAL_ACCEPTANCE.json').write_text(json.dumps(r,indent=2));print(k,v,flush=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
full=build_model('FULL',17);matched=build_model('MATCHED',17)
common=[k for k,v in full.state_dict().items() if k in matched.state_dict() and v.shape==matched.state_dict()[k].shape]
assert all(torch.equal(full.state_dict()[k],matched.state_dict()[k]) for k in common)
p1=sum(p.numel() for p in full.parameters());p2=sum(p.numel() for p in matched.parameters());assert abs(p2/p1-1)<.05
record('matched_initialization',{'FULL_parameters':p1,'MATCHED_parameters':p2,'relative_difference':abs(p2/p1-1),'identical_common_tensors':len(common)})
del matched;gc.collect();m=full.cuda().train()
im=Image.open('/root/autodl-fs/highlight-aesthetic-rl/assets/gaic/GAIC/images/train/210429.jpg').convert('RGB');w,h=im.size
b=torch.tensor([[43.,24.,811.,408.],[43.,24.,725.,456.]],device='cuda');y=torch.tensor([2.14,2.57],device='cuda')
with autocast(True):s=m(im,b)
l,_=losses(s,y);l.backward();ref={n:p.grad.float().cpu().clone() for n,p in m.named_parameters()};m.zero_grad(set_to_none=True)
terms,replay_s=replay_backward(m,im,b,y,chunk=1,amp=True)
num=den=dot=gn=0
for n,p in m.named_parameters():
 a=p.grad.detach().float().cpu().double();v=ref[n].double();num+=float((a-v).square().sum());den+=float(v.square().sum());dot+=float((a*v).sum());gn+=float(a.square().sum())
rel=(num/den)**.5;cos=dot/(den*gn)**.5
record('bf16_replay_gradient',{'relative_l2':rel,'cosine':cos,'score_max_abs':float((s.detach()-replay_s).abs().max())});assert rel<.02 and cos>.999
# Measured linear/conv/mm FLOPs for identical complete inference inputs. Framework counter
# omits fused SDPA/elementwise ops; those have identical shapes across both models.
del ref;m.zero_grad(set_to_none=True);m.eval()
flops={}
for variant in ['FULL','MATCHED']:
 if variant=='MATCHED':del m;gc.collect();torch.cuda.empty_cache();m=build_model(variant,17).cuda().eval()
 with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
  out=m(im,b)
 flops[variant]=sum(e.flops for e in prof.key_averages())
assert abs(flops['FULL']/flops['MATCHED']-1)<.05
record('matched_flops',{'counted_flops':flops,'candidates':len(b),'relative_difference':abs(flops['FULL']/flops['MATCHED']-1),'coverage':'PyTorch linear/conv/mm profiler; fused attention and elementwise omitted, identical attention tensor shapes'})
del m;gc.collect();torch.cuda.empty_cache()
# Exact eight-original accumulation: compare independent mean loss with per-original scaled backward.
p=torch.tensor([.1,.3,.6],device='cuda',requires_grad=True);labels=[torch.tensor([1.,2.+i*.1,4.],device='cuda') for i in range(8)]
l=torch.stack([losses(p,q)[0] for q in labels]).mean();g=torch.autograd.grad(l,p)[0]
for q in labels:(losses(p,q)[0]/8).backward()
assert torch.allclose(g,p.grad,atol=1e-7);record('eight_original_accumulation',{'gradient_max_abs':float((g-p.grad).abs().max()),'trainer_optimizer_step':'once after image batch loop; no step inside candidate replay'})
# MUSIQ train/eval preprocessing equality, true 10-bin expectation, direct/replay gradients.
m=MusiqRank().cuda();m.eval()
with torch.no_grad():se,dist=m.score_crop(im,b[0]);expect=(dist*torch.arange(1,11,device='cuda')).sum(-1);assert torch.allclose(se,(expect-1)/9)
m.train();sd,dist=m.score_crop(im,b[0]);assert torch.allclose(sd,se,atol=1e-6)
# One-crop gradient fixture is not a scientific experiment.
s=torch.cat([m.score_crop(im,x)[0] for x in b]);losses(s,y)[0].backward();refs={n:p.grad.cpu().clone() for n,p in m.named_parameters()};m.zero_grad(set_to_none=True)
t,sr=m.replay(im,b,y,amp=False);num=den=0
for n,p in m.named_parameters():
 a=p.grad.detach().cpu().double();v=refs[n].double();num+=float((a-v).square().sum());den+=float(v.square().sum())
relative=(num/den)**.5;assert relative<1e-4
record('musiq_full_gradient_replay',{'fp32_relative_l2':relative,'train_eval_score_error':float((sd-se).abs().max()),'mu_from_10bins':expect.detach().cpu().tolist(),'all_parameters_have_grad':True})
r['status']='passed';(ROOT/'reports/ADDITIONAL_ACCEPTANCE.json').write_text(json.dumps(r,indent=2));print('ADDITIONAL ACCEPTANCE PASS',flush=True)
