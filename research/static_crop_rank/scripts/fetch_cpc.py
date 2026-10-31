from pathlib import Path
import concurrent.futures as cf,requests,os,re,json,hashlib,tarfile,time,threading
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];base=Path('/root/autodl-fs/highlight-static-rank/data');url='https://drive.google.com/uc?export=download&id=1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO';status=ROOT/'reports/CPC_ACQUISITION.json';guard=threading.Lock();complete=set();started=time.time()
def get(u,**kw):return requests.get(u,timeout=(15,90),**kw)
def write(**kw):status.write_text(json.dumps({'source':url,'elapsed_seconds':time.time()-started,**kw},indent=2))
try:
 r=get(url);r.raise_for_status();form=BeautifulSoup(r.text,'html.parser').find('form');assert form is not None
 params={e['name']:e.get('value','') for e in form.find_all('input') if e.get('name')}
 from urllib.parse import urlencode
 download=form['action']+'?'+urlencode(params)
 r=get(download,headers={'Range':'bytes=0-0'});r.raise_for_status();match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''));assert match and len(r.content)==1
 size=int(match.group(1));assert size<5*1024**3
 target=base/'CPCDataset.tar.gz';part=target.with_suffix('.partial');chunk=4<<20;fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size);count=(size+chunk-1)//chunk;resolved=r.url
 def one(i):
  a=i*chunk;b=min(size,a+chunk)-1
  for attempt in range(3):
   try:
    q=get(resolved,headers={'Range':f'bytes={a}-{b}'});q.raise_for_status();assert q.status_code==206 and q.headers.get('Content-Range')==f'bytes {a}-{b}/{size}' and len(q.content)==b-a+1
    assert os.pwrite(fd,q.content,a)==len(q.content)
    with guard:complete.add(i);write(state='DOWNLOADING',bytes_total=size,chunks_complete=len(complete),chunks_total=count)
    return
   except Exception:
    if attempt==2:raise
 try:
  with cf.ThreadPoolExecutor(8) as pool:list(pool.map(one,range(count)))
  os.fsync(fd)
 finally:os.close(fd)
 assert len(complete)==count
 h=hashlib.sha256()
 with part.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 with tarfile.open(part,'r:gz') as tar:
  members=tar.getmembers();assert sum(x.size for x in members)<12*1024**3
  for member in members:assert not member.issym() and not member.islnk() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  tar.extractall(base/'cpc',filter='data')
 part.replace(target);write(state='ACQUIRED_NOT_YET_PROTOCOL_READY',bytes_total=size,sha256=h.hexdigest(),note='Original overlap audit and native annotation conversion still required; not fed into any training')
except Exception as e:write(state='DOWNLOAD_INCOMPLETE',error=repr(e),completed_chunks=len(complete));raise
