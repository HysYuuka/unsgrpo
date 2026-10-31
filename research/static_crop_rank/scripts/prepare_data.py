from pathlib import Path
import sys,json,hashlib,collections,zipfile
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.core import sha
ROOT=Path(__file__).resolve().parents[1];base=Path('/root/autodl-fs/highlight-static-rank/data');base.mkdir(exist_ok=True,parents=True)
src=Path('/root/autodl-fs/highlight-aesthetic-rl/assets/gaic/GAIC')
archive=src.parents[1]/'GAIC.zip';assert sha(archive)=='b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b'
# Verify extracted bytes against each official ZIP member's CRC, rather than trusting old receipts.
with zipfile.ZipFile(archive) as z:
 import zlib
 for e in z.infolist():
  if e.is_dir():continue
  p=src.parent/e.filename
  assert p.is_file(),str(p)
  assert zlib.crc32(p.read_bytes())==e.CRC,str(p)
rows=[];seen={};pixseen={};errors=[]
for split in ['train','val','test']:
 for p in sorted((src/'images'/split).iterdir()):
  if p.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
  im=Image.open(p).convert('RGB');w,h=im.size
  checksum=sha(p);pixelhash=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
  for table,key in [(seen,checksum),(pixseen,pixelhash)]:
   if key in table:errors.append({'duplicate':str(p),'prior':table[key]})
   table[key]=str(p)
  ann=src/'annotations'/split/(p.stem+'.txt');a=np.loadtxt(ann,ndmin=2)
  assert a.shape[1]==5
  candidates=[]; excluded=[]
  for j,(y1,x1,y2,x2,mos) in enumerate(a.tolist()):
   if mos==-2:
    excluded.append({'native_index':j,'reason':'official sentinel -2; excluded by official GAICD loader'}); continue
   assert np.isfinite([x1,y1,x2,y2,mos]).all() and 0<=x1<x2<=w and 0<=y1<y2<=h and 0<mos<=5,(p,j)
   candidates.append({'crop_id':f'{j:04d}','native_bbox':[y1,x1,y2,x2],'bbox':[x1,y1,x2,y2],'mos':mos})
  rows.append({'source_id':p.stem,'path':str(p),'sha256':checksum,'pixel_sha256':pixelhash,'width':w,'height':h,'split':split,'annotation_sha256':sha(ann),'provenance':'official GAIC journal Google Drive 1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR','n':len(candidates),'native_n':len(a),'excluded_sentinel':excluded,'candidates':candidates})
assert all(Path(e['duplicate']).parent.name==Path(e['prior']).parent.name for e in errors),errors
counts=collections.Counter(r['split'] for r in rows);assert counts=={'train':2636,'val':200,'test':500},counts
official_rows=rows
by_identity={}
for r in rows:
 k=r['pixel_sha256']
 if k in by_identity:
  keep=by_identity[k]; assert r['split']=='train' and keep['split']=='train'
  assert [c['bbox'] for c in keep['candidates']]==[c['bbox'] for c in r['candidates']]
  keep.setdefault('annotation_sets',[[c['mos'] for c in keep['candidates']]]).append([c['mos'] for c in r['candidates']])
  keep.setdefault('source_aliases',[]).append({'source_id':r['source_id'],'path':r['path'],'annotation_sha256':r['annotation_sha256']})
 else:by_identity[k]=r
rows=list(by_identity.values())
manifest=base/'gaic_manifest.jsonl'
manifest.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
report={'images':len(rows),'official_records':len(official_rows),'official_split':dict(counts),'unique_original_split':dict(collections.Counter(r['split'] for r in rows)),'all_candidates':sum(r['n'] for r in rows),'candidates_min':min(r['n'] for r in rows),'candidates_max':max(r['n'] for r in rows),'mos_range':[min(c['mos'] for r in rows for c in r['candidates']),max(c['mos'] for r in rows for c in r['candidates'])],'manifest_sha256':sha(manifest),'official_archive_sha256':sha(archive),'official_split_preserved':True,'duplicate_policy':'One identical train-image pair grouped; same boxes, two published MOS vectors retained; average separately computed annotation-set losses. No relabeling, no candidate deletion.','duplicate_bytes_or_decoded_rgb':errors,'label_policy':'Exclude only official -2 sentinel; retain six published MOS values 0.86/0.93 without clipping; use nominal (MOS-1)/4 affine scale, valid normalized range [-0.035,1]. No inferred or corrected human labels.', 'coordinates':'native y1 x1 y2 x2 reordered to xyxy; pixel-edge convention as official Python crop slicing; no +/-1 shift'}
(ROOT/'reports/DATA_ACCEPTANCE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
