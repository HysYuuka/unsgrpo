"""CPU-only CPC conversion and conservative source overlap audit.

No training modules or GPU runtime are imported. Raw labels and candidate geometry
stay intact; the output is a separate frozen external-inference manifest.
"""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,io,json,re,time
import numpy as np
from scipy.fft import dctn
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/root/autodl-fs/highlight-static-rank/data')
CPC=DATA/'cpc/CPCDataset'
PROTOCOL=ROOT/'configs/cpc_external_protocol.json'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(p,value):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def phash(im):
 a=dctn(np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float),norm='ortho')[:8,:8].flatten()
 flags=a>np.median(a[1:]);flags[0]=False
 return int.from_bytes(np.packbits(flags).tobytes(),'big')

def signature(path):
 raw=Path(path).read_bytes()
 with Image.open(io.BytesIO(raw)) as original:im=original.convert('RGB')
 array=np.asarray(im);mirror=Image.fromarray(np.ascontiguousarray(array[:,::-1]))
 return {'sha256':hashlib.sha256(raw).hexdigest(),'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(),
  'mirrored_pixel_sha256':hashlib.sha256(mirror.tobytes()).hexdigest(),'width':im.width,'height':im.height,
  'phash':phash(im),'phash_mirror':phash(mirror),'thumb':np.asarray(im.resize((32,32),Image.Resampling.BILINEAR))}

def parse(path):
 try:
  raw=path.read_bytes();value=json.loads(raw)
  if isinstance(value,str):value=json.loads(value)
  im=CPC/'images'/path.name[:-4];sig=signature(im)
  boxes=np.asarray(value['bboxes'],dtype=float);scores=np.asarray(value['scores'],dtype=float)
  assert boxes.ndim==2 and boxes.shape[1]==4 and len(boxes)>=5,'candidate_shape'
  assert scores.ndim==2 and scores.shape[1]==len(boxes) and np.isfinite(scores).all(),'rater_shape_or_values'
  assert np.isfinite(boxes).all() and (boxes[:,:2]>=0).all() and (boxes[:,2:]>boxes[:,:2]).all(),'invalid_box'
  assert (boxes[:,[0,2]]<=sig['width']).all() and (boxes[:,[1,3]]<=sig['height']).all(),'outside_image'
  r={'source_id':path.name[:-8],'path':str(im),'split':'external','provenance':'Official CPC raw collected annotations; Google Drive1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO',
   'annotation_path':str(path),'annotation_sha256':hashlib.sha256(raw).hexdigest(),'rater_scores':scores.tolist(),'n':len(boxes),
   'candidates':[{'crop_id':f'{i:04d}','native_bbox':b.tolist(),'bbox':b.tolist(),'mos':float(scores[:,i].mean())} for i,b in enumerate(boxes)],
   'score_semantics':'native assigned preference score averaged over published raters; not 1..5 GAIC MOS'}
  return r,sig,None
 except (OSError,ValueError,KeyError,AssertionError) as e:return None,None,{'annotation':str(path),'reason':type(e).__name__+': '+str(e)}

def main():
 start=time.time();protocol=json.loads(PROTOCOL.read_text());manifest=DATA/'cpc_external_manifest.jsonl';lock=ROOT/'reports/CPC_EXTERNAL_LOCK.json'
 assert not manifest.exists() and not lock.exists(),'Frozen external output already exists; do not silently overwrite'
 assert digest(ROOT/'PROTOCOL_LOCK.json')=='4d75af7657fc42098452b8b7316a9427ea875e136b2059ddaf110d18f5b9537a'
 assert digest(DATA/'gaic_manifest.jsonl')=='0098f7686d7a984b9bfd6993040998a1366c85d5f0a5fcb837a7179404d37d84'
 gaic=[json.loads(l) for l in (DATA/'gaic_manifest.jsonl').read_text().splitlines()]
 rows=[];signatures=[];excluded=[]
 with ThreadPoolExecutor(4) as pool:
  gs=list(pool.map(lambda r:signature(r['path']),gaic))
  for i,(r,sig,error) in enumerate(pool.map(parse,sorted((CPC/'CollectedAnnotationsRaw').glob('*.jpg.txt')))):
   if error:excluded.append(error)
   else:rows.append(r);signatures.append(sig)
   if (i+1)%1000==0:print('CPU parsed originals',i+1,flush=True)
 # Union only source identity; labels and predictions never enter grouping.
 parent=list(range(len(rows)))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):parent[find(a)]=find(b)
 pixels={};families={}
 for i,(r,sig) in enumerate(zip(rows,signatures)):
  family=re.sub(r'_obj_\d+_','_',r['source_id'])
  if family in families:union(i,families[family])
  else:families[family]=i
  for h in (sig['pixel_sha256'],sig['mirrored_pixel_sha256']):
   key=(sig['width'],sig['height'],h)
   if key in pixels:union(i,pixels[key])
   else:pixels[key]=i
 groups=defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 group_name={i:min(rows[j]['source_id'] for j in members) for i,members in groups.items()}
 identity=defaultdict(set)
 for j,sig in enumerate(gs):
  for h in (sig['sha256'],sig['pixel_sha256'],sig['mirrored_pixel_sha256']):identity[h].add(j)
 hashes=np.asarray([sig['phash'] for sig in gs],dtype=np.uint64)
 bits=np.asarray([i.bit_count() for i in range(256)],dtype=np.uint8)
 ratio=np.asarray([s['width']/s['height'] for s in gs])
 ava_ids=defaultdict(list)
 for j,r in enumerate(gaic):ava_ids[r['source_id']].append(j)
 audit=[];blocked=set();test_overlap=set()
 for i,(r,sig) in enumerate(zip(rows,signatures)):
  group=group_name[find(i)];r['source_group']=group
  direct=set().union(*(identity[h] for h in (sig['sha256'],sig['pixel_sha256'],sig['mirrored_pixel_sha256'])))
  dx=hashes^np.uint64(sig['phash']);dm=hashes^np.uint64(sig['phash_mirror'])
  hd=bits[dx.view(np.uint8).reshape(-1,8)].sum(1);hm=bits[dm.view(np.uint8).reshape(-1,8)].sum(1)
  h=np.minimum(hd,hm)
  proposed=set(np.flatnonzero((h<=8)&(np.abs(ratio/(sig['width']/sig['height'])-1)<=.02)).tolist())
  match=re.fullmatch(r'ava_obj_\d+_(\d+)',r['source_id']);name_matches=set(ava_ids.get(match.group(1),[])) if match else set()
  for j in sorted(direct|proposed|name_matches):
   if j in direct:kind='EXACT_BYTES_OR_RGB'
   elif j in proposed:
    image=sig['thumb'][:,::-1] if hm[j]<hd[j] else sig['thumb'];a=image.astype(float).ravel();b=gs[j]['thumb'].astype(float).ravel()
    mae=float(np.abs(a-b).mean());den=np.linalg.norm(a-a.mean())*np.linalg.norm(b-b.mean());corr=float(np.dot(a-a.mean(),b-b.mean())/den) if den else float(np.array_equal(a,b))
    kind='VERIFIED_NEAR_DUPLICATE' if mae<=8 and corr>=.98 else 'AMBIGUOUS_PERCEPTUAL_PROPOSAL'
   else:kind='UNVERIFIED_PUBLISHED_AVA_ID_COLLISION'
   row={'CPC_source':r['source_id'],'CPC_group':group,'GAIC_source':gaic[j]['source_id'],'GAIC_split':gaic[j]['split'],'kind':kind,'hamming':int(h[j])}
   if j in proposed:row.update(RGB_MAE=mae if j not in direct else 0,RGB_correlation=corr if j not in direct else 1)
   audit.append(row)
   if gaic[j]['split'] in ('train','val'):blocked.add(group)
   else:test_overlap.add(group)
  for key in ('sha256','pixel_sha256','mirrored_pixel_sha256','width','height','phash','phash_mirror'):r[key]=sig[key]
  if (i+1)%2000==0:print('CPU overlap screened',i+1,flush=True)
 kept=[r for r in rows if r['source_group'] not in blocked]
 for r in kept:r['overlap_with_GAIC_TEST']=r['source_group'] in test_overlap
 manifest.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in kept))
 (DATA/'cpc_all_native_records.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
 (DATA/'cpc_overlap_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit))
 # Deterministic geometry examples, no learned scores or label-driven selection.
 examples=DATA/'cpc_geometry_examples';examples.mkdir(exist_ok=True);example_paths=[]
 for r in sorted(kept,key=lambda v:hashlib.sha256(v['source_id'].encode()).hexdigest())[:4]:
  with Image.open(r['path']) as source:
   im=source.convert('RGB');canvas=im.copy();draw=ImageDraw.Draw(canvas)
   for j in (0,5,15,23):
    if j>=r['n']:continue
    box=r['candidates'][j]['bbox'];draw.rectangle(box,outline=['red','green','blue','yellow'][[0,5,15,23].index(j)],width=3);draw.text((box[0]+2,box[1]+2),str(j),fill='white')
   canvas.thumbnail((640,640));dest=examples/(r['source_id']+'.jpg');canvas.save(dest);example_paths.append(str(dest))
 report={'time_HK':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),'state':'READY_FOR_FROZEN_CHECKPOINT_EXTERNAL_INFERENCE',
  'CPC_annotated_originals_valid':len(rows),'native_rejected_sources':excluded,'source_groups_before_overlap':len(groups),
  'all_native_candidates_before_overlap':sum(r['n'] for r in rows),'candidate_count_histogram':dict(Counter(r['n'] for r in rows)),
  'raters_per_original_histogram':dict(Counter(len(r['rater_scores']) for r in rows)),
  'raw_assigned_score_range':[min(min(v) for r in rows for v in r['rater_scores']),max(max(v) for r in rows for v in r['rater_scores'])],
  'excluded_TRAIN_VAL_overlap_or_ambiguous_groups':len(blocked),'retained_originals':len(kept),'retained_source_groups':len({r['source_group'] for r in kept}),
  'retained_GAIC_TEST_overlap_groups':len({r['source_group'] for r in kept if r['overlap_with_GAIC_TEST']}),'match_categories':dict(Counter(v['kind'] for v in audit)),
  'manifest_sha256':digest(manifest),'manifest_path':str(manifest),'overlap_audit_sha256':digest(DATA/'cpc_overlap_audit.jsonl'),
  'protocol_sha256':digest(PROTOCOL),'script_sha256':digest(Path(__file__)),'geometry_examples':example_paths,'CPU_wall_seconds':time.time()-start,
  'GPU_used':False,'trained_or_evaluated_models':0,'native_label_scope':'Assigned preference scores from CPC, not GAIC 1..5 MOS; dataset-native metrics remain separate',
  'limitations':'Exact/mirrored/low-distance pHash and conservative AVA-id screen; not a proof that no visually transformed original-content overlap exists. Ambiguous TRAIN/VAL matches excluded without looking at model scores.'}
 write(ROOT/'reports/CPC_CONVERSION_AND_OVERLAP.json',report)
 write(lock,{'status':'FROZEN_BEFORE_CPC_MODEL_INFERENCE','protocol':protocol,'report':report,
  'input_GAIC_manifest_sha256':digest(DATA/'gaic_manifest.jsonl'),'input_CPC_archive_sha256':json.loads((ROOT/'reports/CPC_ACQUISITION.json').read_text())['sha256'],
  'no_CPC_training_or_test_based_tuning':True})
 status=json.loads((ROOT/'reports/EXTERNAL_STATUS.json').read_text());status['CPC'].update(status=report['state'],manifest=str(manifest),protocol='reports/CPC_EXTERNAL_LOCK.json',overlap_audit='reports/CPC_CONVERSION_AND_OVERLAP.json');write(ROOT/'reports/EXTERNAL_STATUS.json',status)
 print(json.dumps({k:v for k,v in report.items() if k not in ('native_rejected_sources','geometry_examples')},indent=2),flush=True)

if __name__=='__main__':main()
