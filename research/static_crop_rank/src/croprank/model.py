"""Full geometry-conditioned scorer. Physical coordinates are pixel edges.

CLIP block equations/state names follow OpenAI CLIP (MIT). Used visual parameters
are loaded strictly; unused global projection/ln_post are explicitly excluded.
"""
from collections import OrderedDict
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18
from .core import image_tensor

WEIGHTS=Path('/root/autodl-fs/highlight-static-rank/weights')
CLIP_NORM=((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
IMAGENET_NORM=((.485,.456,.406),(.229,.224,.225))

def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,c))
class QuickGELU(nn.Module):
 def forward(self,x):return x*torch.sigmoid(1.702*x)
class ClipBlock(nn.Module):
 def __init__(self):
  super().__init__();self.attn=nn.MultiheadAttention(768,12,batch_first=True);self.ln_1=nn.LayerNorm(768);self.ln_2=nn.LayerNorm(768)
  self.mlp=nn.Sequential(OrderedDict([('c_fc',nn.Linear(768,3072)),('gelu',QuickGELU()),('c_proj',nn.Linear(3072,768))]))
 def forward(self,x,invalid):
  z=self.ln_1(x);x=x+self.attn(z,z,z,key_padding_mask=invalid,need_weights=False)[0]
  return x+self.mlp(self.ln_2(x))
class ClipField(nn.Module):
 def __init__(self,pretrained=True):
  super().__init__();self.conv1=nn.Conv2d(3,768,16,stride=16,bias=False)
  self.class_embedding=nn.Parameter(torch.zeros(768));self.positional_embedding=nn.Parameter(torch.zeros(197,768));self.ln_pre=nn.LayerNorm(768)
  self.transformer=nn.Module();self.transformer.resblocks=nn.ModuleList([ClipBlock() for _ in range(12)])
  if pretrained:
   state=torch.jit.load(str(WEIGHTS/'ViT-B-16.pt'),map_location='cpu').state_dict()
   state={k[7:]:v.float() for k,v in state.items() if k.startswith('visual.') and k not in ['visual.proj','visual.ln_post.weight','visual.ln_post.bias']}
   self.load_state_dict(state,strict=True)
  self.checkpoint=True
 def forward(self,x):
  x=self.conv1(x);h,w=x.shape[-2:];x=x.flatten(2).transpose(1,2)
  cls=self.class_embedding[None,None].expand(1,1,-1);x=torch.cat([cls,x],1)
  pos=F.interpolate(self.positional_embedding[1:].T.reshape(1,768,14,14),size=(h,w),mode='bicubic',align_corners=False).flatten(2).transpose(1,2)
  x=self.ln_pre(x+torch.cat([self.positional_embedding[:1][None],pos],1))
  # Padding is <16 pixels, so every produced patch intersects the valid image.
  # Padded input is normalized zero; spatial area is accounted for in attention.
  invalid=torch.zeros((1,1+h*w),dtype=torch.bool,device=x.device)
  outputs=[]
  for i,block in enumerate(self.transformer.resblocks):
   x=checkpoint(block,x,invalid,use_reentrant=False) if self.checkpoint and self.training and torch.is_grad_enabled() else block(x,invalid)
   if i in [3,7,11]:outputs.append(x[:,1:])
  return outputs,h,w

class DetailField(nn.Module):
 def __init__(self,pretrained=True):
  super().__init__();r=resnet18(weights=None)
  if pretrained:r.load_state_dict(torch.load(WEIGHTS/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True),strict=True)
  self.stem=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool);self.layer1=r.layer1;self.layer2=r.layer2
  self.p4=nn.Conv2d(64,128,1);self.p8=nn.Conv2d(128,128,1)
 def train(self,mode=True):
  super().train(mode)
  for m in self.modules():
   if isinstance(m,nn.BatchNorm2d):m.eval()
  return self
 def forward(self,x):
  d4=self.layer1(self.stem(x));d8=self.layer2(d4)
  d4=self.p4(d4);d8=self.p8(d8)
  h,w=d4.shape[-2:];y,x=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
  # Both convolution grids start at pixel center 0.5; q8=q4/2.
  grid=torch.stack([2*(x/2+.5)/d8.shape[-1]-1,2*(y/2+.5)/d8.shape[-2]-1],-1)[None]
  aligned=F.grid_sample(d8,grid.to(d8.dtype),mode='bilinear',padding_mode='border',align_corners=False)
  return torch.cat([d4,aligned],1)

class Coord(nn.Module):
 def __init__(self,crop=False):
  super().__init__();self.crop=crop;self.net=mlp(22 if crop else 18,128,256)
 def forward(self,x):
  flags=torch.cat([(x< -4).to(x),(x>4).to(x)],-1) if self.crop else None
  if self.crop:x=x.clamp(-4,4)
  freq=x.new_tensor([1,2,4,8]);angles=2*math.pi*x[...,None]*freq
  z=torch.cat([x,angles.sin().flatten(-2),angles.cos().flatten(-2)],-1)
  if self.crop:z=torch.cat([z,flags],-1)
  return self.net(z)

def sample_field(field,xy,meta,stride,center):
 """xy (...,2) original edges -> exact feature center indices."""
 hw=field.shape[-2:];scale=xy.new_tensor(meta['scale'])
 q=(xy*scale-center)/stride
 grid=2*(q+.5)/xy.new_tensor([hw[1],hw[0]])-1
 shape=xy.shape[:-1];grid=grid.reshape(1,-1,1,2).to(field.dtype)
 out=F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)
 return out[0,:,:,0].T.reshape(*shape,field.shape[1])

def grid_xy(h,w,meta,stride,center,device):
 yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
 return (torch.stack([xx,yy],-1).float()*stride+center)/torch.tensor(meta['scale'],device=device)

def coverage(boxes,meta,h,w):
 device=boxes.device;sy,sx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
 lo=torch.stack([sx,sy],-1).float().reshape(-1,2)*16/boxes.new_tensor(meta['scale'])
 hi=torch.minimum(lo+16/boxes.new_tensor(meta['scale']),boxes.new_tensor(meta['size']))
 area=(hi-lo).clamp_min(0).prod(-1)
 intersection=(torch.minimum(hi[None],boxes[:,None,2:])-torch.maximum(lo[None],boxes[:,None,:2])).clamp_min(0).prod(-1)
 alpha=intersection/area.clamp_min(1e-12)
 valid_fraction=area/(256/boxes.new_tensor(meta['scale']).prod())
 return alpha,valid_fraction

def boundary_points(boxes):
 t=(torch.arange(16,device=boxes.device,dtype=boxes.dtype)+.5)/16
 low,high=boxes[:,:2],boxes[:,2:];wh=high-low
 x=low[:,0,None]+wh[:,0,None]*t;y=low[:,1,None]+wh[:,1,None]*t
 points=torch.stack([torch.stack([x,low[:,1,None].expand_as(x)],-1),torch.stack([high[:,0,None].expand_as(y),y],-1),torch.stack([x,high[:,1,None].expand_as(x)],-1),torch.stack([low[:,0,None].expand_as(y),y],-1)],1).reshape(-1,64,2)
 inward=boxes.new_tensor([[0,1],[-1,0],[0,-1],[1,0]]).repeat_interleave(16,dim=0)
 offset=.04*wh.min(-1).values[:,None,None]
 return torch.stack([points+offset*inward,points,points-offset*inward],2),points

class Cross(nn.Module):
 def __init__(self):
  super().__init__();self.q=nn.Linear(256,256);self.k=nn.Linear(256,256);self.v=nn.Linear(256,256);self.out=nn.Linear(256,256)
 def forward(self,q,src,weights):
  b,n,_=q.shape;m=src.shape[1]
  Q=self.q(q).reshape(b,n,8,32).transpose(1,2);K=self.k(src).reshape(b,m,8,32).transpose(1,2);V=self.v(src).reshape(b,m,8,32).transpose(1,2)
  empty=weights.sum(-1)==0
  safe=torch.where(empty[:,None],torch.ones_like(weights),weights)
  mask=safe.float().clamp_min(1e-30).log().masked_fill(safe==0,float('-inf'))[:,None,None,:].to(Q.dtype)
  z=F.scaled_dot_product_attention(Q,K,V,attn_mask=mask).transpose(1,2).reshape(b,n,256)
  return self.out(z).masked_fill(empty[:,None,None],0),empty

class CropRank(nn.Module):
 def __init__(self,variant='FULL',pretrained=True):
  super().__init__();self.variant=variant;self.explicit=variant not in ('MATCHED','MATCHED_BASE');self.boundary=variant!='NO_BOUNDARY'
  self.semantic=ClipField(pretrained);self.detail=DetailField(pretrained)
  self.sem_projs=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256)) for _ in range(3)]);self.layer_logits=nn.Parameter(torch.zeros(3))
  self.source_coord=Coord();self.crop_coord=Coord(crop=True);self.bbox=mlp(8,128,256);self.layout_project=nn.Linear(512,256)
  self.inside=Cross();self.outside=Cross();self.gate=mlp(1024,256,512);self.empty_embed=nn.Linear(2,256,bias=False)
  self.relation_norm=nn.LayerNorm(256);self.relation_ff=mlp(256,1024,256)
  # MATCHED retains all raw samples and conventional anchor positions. No signed difference.
  boundary_in=(4 if self.explicit else 3)*256+3+4+256+256
  if self.boundary:self.boundary_mlp=mlp(boundary_in,256,256)
  self.score_token=nn.Parameter(torch.zeros(1,1,256))
  layer=nn.TransformerEncoderLayer(256,8,1024,dropout=0,activation='gelu',batch_first=True,norm_first=True)
  self.fusion=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
  self.score=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
 def encode(self,im):
  device=next(self.parameters()).device
  sx,sm=image_tensor(im,512,device,CLIP_NORM,16);dx,dm=image_tensor(im,1024,device,IMAGENET_NORM)
  feats,h,w=self.semantic(sx)
  sem=sum(a*p(f) for a,p,f in zip(self.layer_logits.softmax(0),self.sem_projs,feats)).transpose(1,2).reshape(1,256,h,w)
  detail=checkpoint(self.detail,dx,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.detail(dx)
  return (sem,detail),{'semantic':sm,'detail':dm}
 def score_fields(self,fields,meta,boxes,debug=False):
  sem,detail=fields;sm,dm=meta['semantic'],meta['detail'];b=len(boxes);wh=boxes[:,2:]-boxes[:,:2];size=boxes.new_tensor(sm['size'])
  bf=torch.cat([boxes/size.repeat(2),wh/size,(wh.prod(-1)/size.prod())[:,None],(wh[:,0]/wh[:,1]).log()[:,None]],-1)
  be=self.bbox(bf)
  t=(torch.arange(9,device=boxes.device,dtype=boxes.dtype)+.5)/9;yy,xx=torch.meshgrid(t,t,indexing='ij');local=torch.stack([xx,yy],-1).reshape(1,81,2).expand(b,-1,-1)
  xy=boxes[:,None,:2]+local*wh[:,None]
  q=self.layout_project(torch.cat([sample_field(sem,xy,sm,16,8),sample_field(detail,xy,dm,4,.5)],-1))
  q=q+self.source_coord(xy/size)+self.crop_coord(local)+be[:,None]
  source_xy=grid_xy(*sem.shape[-2:],sm,16,8,boxes.device).reshape(1,-1,2)
  source=sem.flatten(2).transpose(1,2)+self.source_coord(source_xy/size)
  source=source.expand(b,-1,-1)+be[:,None]
  if self.explicit:source=source+self.crop_coord((source_xy-boxes[:,None,:2])/wh[:,None])
  alpha,area=coverage(boxes,sm,*sem.shape[-2:]);rin,ein=self.inside(q,source,alpha*area);rout,eout=self.outside(q,source,(1-alpha)*area)
  logits=self.gate(torch.cat([q,rin,rout,be[:,None].expand(-1,81,-1)],-1)).reshape(b,81,2,256)
  empties=torch.stack([ein,eout],-1);logits=logits.masked_fill(empties[:,None,:,None],float('-inf'));g=logits.softmax(2)
  q=q+g[:,:,0]*rin+g[:,:,1]*rout+self.empty_embed(empties.float())[:,None]
  q=q+self.relation_ff(self.relation_norm(q))
  if self.boundary:
   bp,anchors=boundary_points(boxes)
   valid=((bp>=0)&(bp<=size)).all(-1)
   samples=sample_field(detail,bp,dm,4,.5)*valid[...,None]
   parts=[samples.flatten(-2)]
   if self.explicit:parts.append(samples[:,:,0]-samples[:,:,2])
   edgeid=F.one_hot(torch.arange(4,device=boxes.device).repeat_interleave(16),4).float()[None].expand(b,-1,-1)
   c=(anchors-boxes[:,None,:2])/wh[:,None]
   parts.extend([valid.float(),edgeid,self.source_coord(anchors/size),self.crop_coord(c)])
   bt=self.boundary_mlp(torch.cat(parts,-1));bv=valid[:,:,1]
   tokens=torch.cat([self.score_token.expand(b,-1,-1),q,bt],1)
   mask=torch.cat([torch.zeros(b,82,dtype=torch.bool,device=boxes.device),~bv],1)
  else:
   tokens=torch.cat([self.score_token.expand(b,-1,-1),q],1);mask=None
  out=self.fusion(tokens,src_key_padding_mask=mask)
  bm=(out[:,82:]*bv[...,None]).sum(1)/bv.sum(1).clamp_min(1)[:,None] if self.boundary else torch.zeros_like(be)
  score=self.score(torch.cat([out[:,0],out[:,1:82].mean(1),bm,be],-1)).flatten()
  if debug:return score,{'source':source,'alpha':alpha,'empty':empties,'boundary':bt if self.boundary else None,'layout':q,'meta':meta}
  return score
 def forward(self,im,boxes):
  fields,meta=self.encode(im);return self.score_fields(fields,meta,boxes)
