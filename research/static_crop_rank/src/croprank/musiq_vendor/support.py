import torch
from pathlib import Path
def dist_to_mos(x):return (x*torch.arange(1,x.shape[-1]+1,device=x.device,dtype=x.dtype)).sum(-1,keepdim=True)
def get_url_from_name(x):return str(Path('/root/autodl-fs/highlight-static-rank/weights')/x)
def load_pretrained_network(net,path,strict=True):
 state=torch.load(path,map_location='cpu',weights_only=True)
 if 'params' in state:state=state['params']
 state={k.removeprefix('module.'):v for k,v in state.items()}
 net.load_state_dict(state,strict=True)
