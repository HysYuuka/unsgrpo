from contextlib import nullcontext
import torch
from .core import losses

def autocast(enabled):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)

def replay_backward(model,im,boxes,mos,chunk=4,scale=1.,selection=True,amp=True):
 with autocast(amp):originals,meta=model.encode(im)
 leaves=tuple(x.detach().requires_grad_() for x in originals)
 states=[];scores=[]
 with torch.no_grad():
  for b in boxes.split(chunk):
   states.append((torch.get_rng_state(),torch.cuda.get_rng_state()))
   with autocast(amp):scores.append(model.score_fields(leaves,meta,b).float())
 final_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
 s=torch.cat(scores).detach().requires_grad_();loss,terms=losses(s,mos,selection)
 (v,)=torch.autograd.grad(loss*scale,s)
 for i,b in enumerate(boxes.split(chunk)):
  torch.set_rng_state(states[i][0]);torch.cuda.set_rng_state(states[i][1])
  with autocast(amp):out=model.score_fields(leaves,meta,b)
  (out.float()*v[i*chunk:i*chunk+len(b)]).sum().backward()
 torch.set_rng_state(final_rng[0]);torch.cuda.set_rng_state(final_rng[1])
 torch.autograd.backward(originals,[x.grad for x in leaves])
 return {k:float(z.detach()) for k,z in terms.items()},s.detach()

@torch.no_grad()
def predict(model,im,boxes,chunk=4,amp=True):
 with autocast(amp):
  fields,meta=model.encode(im)
  return torch.cat([model.score_fields(fields,meta,b).float() for b in boxes.split(chunk)])
