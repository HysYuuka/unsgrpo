from pathlib import Path
import urllib.request,hashlib,json,time,concurrent.futures
ROOT=Path('/root/autodl-fs/highlight-static-rank/weights')
def get(name,url,expected):
 out=ROOT/name;part=ROOT/(name+'.partial')
 if out.exists():return
 errors=[]
 for proxy in [False,True]:
  try:
   op=urllib.request.build_opener(*([] if proxy else [urllib.request.ProxyHandler({})]))
   with op.open(url,timeout=30) as r,part.open('wb') as f:
    h=hashlib.sha256();n=0
    while True:
     b=r.read(1<<20)
     if not b:break
     f.write(b);h.update(b);n+=len(b)
   sha=h.hexdigest()
   if not sha.startswith(expected):raise ValueError('SHA mismatch '+sha)
   part.replace(out);(ROOT/(name+'.receipt.json')).write_text(json.dumps({'source':url,'bytes':n,'sha256':sha,'time':time.time()},indent=2));print(name,n,sha,flush=True);return
  except Exception as e:errors.append(repr(e))
 raise RuntimeError(name+str(errors))
assets=[('ViT-B-16.pt','https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt','5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'),('resnet18-f37072fd.pth','https://download.pytorch.org/models/resnet18-f37072fd.pth','f37072fd'),('musiq_ava_ckpt-e8d3f067.pth','https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_ava_ckpt-e8d3f067.pth','e8d3f067')]
with concurrent.futures.ThreadPoolExecutor(3) as e:
 for f in [e.submit(get,*a) for a in assets]:
  try:f.result()
  except Exception as ex: print(ex,flush=True)
