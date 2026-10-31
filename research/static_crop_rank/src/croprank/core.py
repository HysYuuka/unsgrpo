from pathlib import Path
import hashlib, json, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.stats import spearmanr

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def keyed_seed(*items): return int(hashlib.sha256(':'.join(map(str,items)).encode()).hexdigest()[:15],16)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def losses(scores, mos, selection=True):
 if mos.ndim==2:
  results=[losses(scores,m,selection) for m in mos]
  return torch.stack([r[0] for r in results]).mean(),{k:torch.stack([r[1][k] for r in results]).mean() for k in results[0][1]}
 s=scores.float(); y=(mos.float()-1)/4
 reg=F.smooth_l1_loss(s,y,beta=.1)
 i,j=torch.triu_indices(len(s),len(s),1,device=s.device)
 gap=y[i]-y[j]; w=(gap.abs()/.05).clamp(max=1)
 rank=(w*F.binary_cross_entropy_with_logits((s[i]-s[j])/.1,torch.sigmoid(gap/.1),reduction='none')).sum()/w.sum().clamp_min(1e-12)
 sel=(F.softmax(s/.1,dim=0)*(y.max()-y-.05).clamp_min(0)).sum()
 total=reg+.2*rank+(sel if selection else 0)
 return total, {'reg':reg,'rank':rank,'sel':sel,'total':total}

def metrics(scores,mos):
 s=np.asarray(scores,dtype=float);y=np.asarray(mos,dtype=float)
 assert len(s)==len(y) and len(s)>0 and np.isfinite(s).all() and np.isfinite(y).all()
 order=np.argsort(-s,kind='stable'); gy=np.sort(y)[::-1]
 out={'regret':float(gy[0]-y[order[0]]),'srcc':float(spearmanr(s,y).statistic) if np.ptp(s)>0 and np.ptp(y)>0 else 0.0,'srcc_degenerate':int(np.ptp(s)==0 or np.ptp(y)==0)}
 for n in (5,10):
  threshold=gy[min(n,len(y))-1]
  for k in range(1,5):out[f'acc{k}_{n}']=float((y[order[:k]]>=threshold).mean())
  out[f'mean_acc_{n}']=float(np.mean([out[f'acc{k}_{n}'] for k in range(1,5)]))
 return out

def load_records(manifest,split):
 return [json.loads(l) for l in Path(manifest).read_text().splitlines() if json.loads(l)['split']==split]

def load_image(record,seed=0,epoch=0,augment=False):
 im=Image.open(record['path']).convert('RGB')
 boxes=torch.tensor([x['bbox'] for x in record['candidates']],dtype=torch.float32)
 mos=torch.tensor(record.get('annotation_sets',[x['mos'] for x in record['candidates']]),dtype=torch.float32)
 flip=augment and keyed_seed(seed,epoch,record['source_id'])%2==1
 if flip:
  im=ImageOps.mirror(im); boxes[:,[0,2]]=im.width-boxes[:,[2,0]]
 return im,boxes,mos

def image_tensor(im,long_edge,device,normalization,pad=1):
 w,h=im.size;ratio=long_edge/max(w,h); nw,nh=round(w*ratio),round(h*ratio)
 resized=im.resize((nw,nh),Image.Resampling.BICUBIC)
 x=torch.from_numpy(np.array(resized,copy=True)).permute(2,0,1).float().div_(255).to(device)[None]
 means,stds=normalization
 x=(x-x.new_tensor(means)[None,:,None,None])/x.new_tensor(stds)[None,:,None,None]
 x=F.pad(x,(0,(-nw)%pad,0,(-nh)%pad))
 return x, {'size':(w,h),'resized':(nw,nh),'scale':(nw/w,nh/h),'long_edge':long_edge}
