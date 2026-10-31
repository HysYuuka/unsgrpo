from pathlib import Path
import sys,json,gc,time
import numpy as np,torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.musiq import MusiqRank
from croprank.core import losses,load_records,load_image
ROOT=Path(__file__).resolve().parents[1];device=sys.argv[1] if len(sys.argv)>1 else 'cpu';torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=MusiqRank().to(device).train();records=load_records('/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl','train');im,b,y=load_image(records[0]);b=b[:4].to(device);y=y[:4].to(device)
ref=torch.cat([m.score_crop(im,x)[0] for x in b]);losses(ref,y)[0].backward();grads={n:p.grad.cpu().clone() for n,p in m.named_parameters()};m.zero_grad(set_to_none=True)
s,_=m.score_crops(im,b);losses(s,y)[0].backward();num=den=0.;worst=0.
for n,p in m.named_parameters():
 a=p.grad.detach().cpu().double();g=grads[n].double();u=float((a-g).square().sum());v=float(g.square().sum());num+=u;den+=v
 if v>1e-12:worst=max(worst,(u/v)**.5)
r={'status':'passed','device':device,'chunk':4,'full_native_preprocessing':True,'additional_padding_mask':0,'score_max_abs':float((s-ref).detach().abs().max()),'gradient_relative_l2':(num/den)**.5,'worst_material_tensor':worst,'all_parameters_checked':len(grads)}
assert r['score_max_abs']<1e-5 and r['gradient_relative_l2']<1e-4 and worst<1e-4,r
(ROOT/'reports/MUSIQ_BATCH_ACCEPTANCE.json').write_text(json.dumps(r,indent=2));print(r)
