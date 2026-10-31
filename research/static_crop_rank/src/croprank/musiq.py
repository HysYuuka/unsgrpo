import torch
from torch import nn
import numpy as np
from .musiq_vendor.model import MUSIQ
from .replay import autocast
from .core import losses
from .musiq_vendor.multiscale import get_multiscale_patches
import torch.nn.functional as F

class MusiqRank(nn.Module):
 def __init__(self):
  super().__init__();self.network=MUSIQ(num_class=10,pretrained=False,pretrained_model_path='/root/autodl-fs/highlight-static-rank/weights/musiq_ava_ckpt-e8d3f067.pth')
 def score_crop(self,im,box):
  # Native integer GAIC boundaries; preserve original crop resolution and all patches.
  vals=box.detach().cpu().tolist();assert all(v==int(v) for v in vals)
  crop=im.crop(tuple(map(int,vals)))
  x=torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()[None].to(next(self.parameters()).device)/255
  mu,dist=self.network(x,return_dist=True)
  return ((mu-1)/9).flatten(),dist
 def score_crops(self,im,boxes):
  patches=[]
  for box in boxes:
   vals=box.detach().cpu().tolist();assert all(v==int(v) for v in vals)
   crop=im.crop(tuple(map(int,vals)))
   x=torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()[None].to(next(self.parameters()).device)/255
   patches.append(get_multiscale_patches((x-.5)*2,**self.network.data_preprocess_opts))
  n=max(x.shape[1] for x in patches)
  # Additional padding has validity mask 0, not additional image evidence.
  x=torch.cat([F.pad(p,(0,0,0,n-p.shape[1])) for p in patches],0)
  mu,dist=self.network(x,return_dist=True)
  return ((mu-1)/9).flatten(),dist
 def predict(self,im,boxes,amp=True):
  with torch.no_grad(),autocast(amp):return torch.cat([self.score_crops(im,b)[0].float() for b in boxes.split(4)])
 def replay(self,im,boxes,mos,scale=1.,selection=True,amp=True):
  rng=[];scores=[];chunks=list(boxes.split(4))
  with torch.no_grad(),autocast(amp):
   for b in chunks:
    rng.append((torch.get_rng_state(),torch.cuda.get_rng_state()));scores.append(self.score_crops(im,b)[0].float())
  end=(torch.get_rng_state(),torch.cuda.get_rng_state())
  s=torch.cat(scores).detach().requires_grad_();loss,terms=losses(s,mos,selection);(v,)=torch.autograd.grad(loss*scale,s)
  offset=0
  for i,b in enumerate(chunks):
   torch.set_rng_state(rng[i][0]);torch.cuda.set_rng_state(rng[i][1])
   with autocast(amp):out=self.score_crops(im,b)[0]
   (out.float()*v[offset:offset+len(b)]).sum().backward();offset+=len(b)
  torch.set_rng_state(end[0]);torch.cuda.set_rng_state(end[1])
  return {k:float(z.detach()) for k,z in terms.items()},s.detach()
