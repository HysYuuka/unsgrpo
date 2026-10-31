"""Official GAIC journal MobileNet weights, exact forward-kernel PyTorch port.

The legacy custom ROI/RoD kernels use sampling/extrapolation conventions different
from torchvision.roi_align. Preserve their published arithmetic explicitly.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .gaic_mobilenet import MobileNetV2

def kernel_align(feat,boxes,discard=False):
 _,c,h,w=feat.shape;b=len(boxes);box=boxes/16
 t=torch.arange(10,device=feat.device,dtype=feat.dtype)/9
 if discard:
  yy,xx=torch.meshgrid(t*(h-1.001),t*(w-1.001),indexing='ij');x=xx[None].expand(b,-1,-1);y=yy[None].expand(b,-1,-1)
  valid=~((x>=box[:,0,None,None])&(x<=box[:,2,None,None])&(y>=box[:,1,None,None])&(y<=box[:,3,None,None]))
 else:
  x=box[:,0,None]+(box[:,2]-box[:,0]+1).clamp_min(0)[:,None]*t;y=box[:,1,None]+(box[:,3]-box[:,1]+1).clamp_min(0)[:,None]*t
  x=x[:,None,:].expand(-1,10,-1);y=y[:,:,None].expand(-1,-1,10);valid=(x>=0)&(x<w)&(y>=0)&(y<h)
 ix=x.floor().long().clamp(0,w-2);iy=y.floor().long().clamp(0,h-2);dx=x-ix;dy=y-iy
 base=feat[0].permute(1,2,0)
 z=base[iy,ix]*(1-dx)[...,None]*(1-dy)[...,None]+base[iy,ix+1]*dx[...,None]*(1-dy)[...,None]+base[iy+1,ix]*(1-dx)[...,None]*dy[...,None]+base[iy+1,ix+1]*dx[...,None]*dy[...,None]
 return F.avg_pool2d((z*valid[...,None]).permute(0,3,1,2),2,1)

class GAIC(nn.Module):
 def __init__(self):
  super().__init__();m=MobileNetV2();self.Feat_ext=nn.Module();self.Feat_ext.feature3=nn.Sequential(m.features[:7]);self.Feat_ext.feature4=nn.Sequential(m.features[7:14]);self.Feat_ext.feature5=nn.Sequential(m.features[14:]);self.DimRed=nn.Conv2d(448,8,1)
  self.FC_layers=nn.Sequential(nn.Sequential(nn.Conv2d(16,768,9),nn.ReLU(inplace=True)),nn.Sequential(nn.Conv2d(768,128,1),nn.ReLU(inplace=True)),nn.Conv2d(128,1,1))
  weights=next(Path('/root/autodl-fs/highlight-static-rank/weights/gaic_official').glob('mobilenet_*.pth'));self.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
 def forward(self,x,boxes):
  f3=self.Feat_ext.feature3(x);f4=self.Feat_ext.feature4(f3);f5=self.Feat_ext.feature5(f4)
  cat=torch.cat([F.interpolate(f3,scale_factor=.5,mode='bilinear',align_corners=True),f4,.5*F.interpolate(f5,scale_factor=2,mode='bilinear',align_corners=True)],1)
  f=self.DimRed(cat);return self.FC_layers(torch.cat([kernel_align(f,boxes),kernel_align(f,boxes,True)],1)).flatten()
 def predict(self,im,boxes):
  import cv2
  image=np.array(im);h,w=image.shape[:2];scale=256/min(h,w);nh=round(h*scale/32)*32;nw=round(w*scale/32)*32
  x=cv2.resize(image,(nw,nh)).astype(np.float32)/256
  x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  x=torch.from_numpy(x).permute(2,0,1)[None].to(next(self.parameters()).device);b=boxes.clone();b[:,[0,2]]*=nw/w;b[:,[1,3]]*=nh/h;b[:,:2]=b[:,:2].floor();b[:,2:]=b[:,2:].ceil()
  return self(x,b)
