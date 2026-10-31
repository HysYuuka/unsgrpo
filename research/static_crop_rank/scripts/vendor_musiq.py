from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parents[1];out=root/'src/croprank/musiq_vendor';out.mkdir(exist_ok=True)
z=zipfile.ZipFile(next((root/'vendor').glob('pyiqa*.whl')))
(out/'__init__.py').write_text('')
for orig,name in [('pyiqa/matlab_utils/padding.py','padding.py'),('pyiqa/data/multiscale_trans_util.py','multiscale.py')]:
 (out/name).write_bytes(z.read(orig))
s=z.read('pyiqa/archs/musiq_arch.py').decode()
s=s.replace('from .arch_util import dist_to_mos, load_pretrained_network','from .support import dist_to_mos, load_pretrained_network')
s=s.replace('from pyiqa.matlab_utils import ExactPadding2d, exact_padding_2d','from .padding import ExactPadding2d, exact_padding_2d')
s=s.replace('from pyiqa.utils.registry import ARCH_REGISTRY','')
s=s.replace('from pyiqa.data.multiscale_trans_util import get_multiscale_patches','from .multiscale import get_multiscale_patches')
s=s.replace('from pyiqa.archs.arch_util import get_url_from_name','from .support import get_url_from_name').replace('@ARCH_REGISTRY.register()','')
s=s.replace('if not self.training:', 'if x.ndim == 4 and x.shape[1] == 3:')
(out/'model.py').write_text(s)
(out/'support.py').write_text('''import torch
from pathlib import Path
def dist_to_mos(x):return (x*torch.arange(1,x.shape[-1]+1,device=x.device,dtype=x.dtype)).sum(-1,keepdim=True)
def get_url_from_name(x):return str(Path('/root/autodl-fs/highlight-static-rank/weights')/x)
def load_pretrained_network(net,path,strict=True):
 state=torch.load(path,map_location='cpu',weights_only=True)
 if 'params' in state:state=state['params']
 state={k.removeprefix('module.'):v for k,v in state.items()}
 net.load_state_dict(state,strict=True)
''')
for n in z.namelist():
 if n.lower().endswith('license'): (out/'LICENSE').write_bytes(z.read(n))
(root/'reports/MUSIQ_VENDOR_ADAPTATION.json').write_text(json.dumps({'source':'pyiqa 0.1.16 wheel','wheel_sha256':hashlib.sha256(Path(z.filename).read_bytes()).hexdigest(),'changes':['Isolate official network, padding, multiscale modules; remove global registry/import unrelated architectures','Strict local weight load; 10-bin MOS unchanged','Preprocess RGB inputs in both train and eval; original upstream only preprocesses eval and expects preprocessed training patches'],'architecture_changed':False,'native_scales':[224,384],'original_resolution_patch_limit':-1},indent=2))
