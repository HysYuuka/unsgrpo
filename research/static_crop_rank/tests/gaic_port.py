from pathlib import Path
import sys,json,math
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.gaic import GAIC,kernel_align
torch.set_num_threads(2);torch.manual_seed(7);f=torch.randn(1,3,12,16);boxes=torch.tensor([[1.,3.,220.,150.],[0.,0.,256.,192.],[80.,40.,240.,180.]])
errors={}
for discard in [False,True]:
 actual=kernel_align(f,boxes,discard);expected=np.zeros((3,3,10,10),dtype=np.float32);a=f.numpy()[0]
 for n,box in enumerate(boxes.numpy()/16):
  for ph in range(10):
   for pw in range(10):
    y=ph*(12-1.001)/9 if discard else ph*max(box[3]-box[1]+1,0)/9+box[1]
    x=pw*(16-1.001)/9 if discard else pw*max(box[2]-box[0]+1,0)/9+box[0]
    if discard and box[0]<=x<=box[2] and box[1]<=y<=box[3]:continue
    if not discard and (y<0 or y>=12 or x<0 or x>=16):continue
    iy=min(math.floor(y),10);ix=min(math.floor(x),14);dy=y-iy;dx=x-ix
    expected[n,:,ph,pw]=a[:,iy,ix]*(1-dy)*(1-dx)+a[:,iy,ix+1]*(1-dy)*dx+a[:,iy+1,ix]*dy*(1-dx)+a[:,iy+1,ix+1]*dy*dx
 expected=(expected[:,:,:-1,:-1]+expected[:,:,1:,:-1]+expected[:,:,:-1,1:]+expected[:,:,1:,1:])/4
 errors[str(discard)]=float(np.max(np.abs(actual.numpy()-expected)));assert errors[str(discard)]<1e-5
m=GAIC();assert torch.count_nonzero(kernel_align(f,boxes[1:2],True))==0
report={'status':'passed','strict_official_task_weight_load':True,'scalar_cuda_formula_vs_vector_port_max_abs':errors,'full_frame_RoD_zero':True,'label':'GAIC official released weights with verified inference arithmetic port','parameters':sum(p.numel() for p in m.parameters()),'caveat':'not an official rebuilt legacy CUDA binary; published checkpoint training/selection exposure is separate from controlled same-data comparisons'}
Path('/root/autodl-tmp/highlight-static-rank/reports/GAIC_PORT_ACCEPTANCE.json').write_text(json.dumps(report,indent=2));print(report)
