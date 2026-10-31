from pathlib import Path
import urllib.request,json,concurrent.futures,time
urls=['https://hf-mirror.com/api/models/yzc002/FGAesQ/tree/main','https://hf-mirror.com/api/datasets/BWGZK/procrop_dataset/tree/main','https://hf-mirror.com/api/datasets/yzc002/FGAesthetics/tree/main']
def one(u):
 for proxy in [False,True]:
  try:
   op=urllib.request.build_opener(*([] if proxy else [urllib.request.ProxyHandler({})]));d=json.load(op.open(u,timeout=12));return {'url':u,'data':d}
  except Exception as e:err=repr(e)
 return {'url':u,'error':err}
with concurrent.futures.ThreadPoolExecutor(3) as ex:results=list(ex.map(one,urls))
Path('/root/autodl-tmp/highlight-static-rank/reports/EXTERNAL_MIRROR_LOOKUP.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
