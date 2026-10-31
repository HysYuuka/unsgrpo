#!/usr/bin/env python3
"""Review-only dense frames for all six fixed cases; never model inputs."""
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import av
from unsgrpo.round2_utils import *


def main():
    start=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();cases=read(ROOT/'teachers/cases.json')['cases']
    out=ROOT/'visual_review';out.mkdir(exist_ok=False);receipts=[]
    for number,case in enumerate(cases):
        k=case['video_id'];a,b=case['bounds'];source=Path(by[k]['path'])
        if sha(source)!=by[k]['raw_sha256']:raise ValueError('Original video changed')
        targets=np.arange(a,b,.125);images=[];pts=[];n=0
        with av.open(str(source)) as container:
            stream=container.streams.video[0];start_pts=float((stream.start_time or 0)*stream.time_base)
            container.seek(int((a+start_pts)/float(stream.time_base)),stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:continue
                t=float(frame.pts*frame.time_base)-start_pts
                while n<len(targets) and t+1e-9>=targets[n]:
                    image=frame.to_image();image.thumbnail((320,180));images.append(image.copy());pts.append(t);n+=1
                if n==len(targets):break
        if len(images)!=16:raise ValueError('Dense review failed to cover selected two-second clip')
        sheet=Image.new('RGB',(1280,800),'white');draw=ImageDraw.Draw(sheet)
        for j,(im,t) in enumerate(zip(images,pts)):
            x=(j%4)*320;y=(j//4)*200;sheet.paste(im,(x+(320-im.width)//2,y));draw.text((x+4,y+183),f'{t:.3f}s REVIEW ONLY 8fps',fill='black')
        path=out/f'case{number+1}_{k}_{a:g}s_review_only.jpg';sheet.save(path,quality=90)
        receipts.append({'video_id':k,'bounds':case['bounds'],'source':reference(source),'actual_pts':pts,
            'review_sheet':reference(path),'frames_fed_to_any_model':0,'sampling':'First native PTS after target at 8Hz, review only; original four model frames unchanged'})
    write(out/'dense_frame_review_receipt.json',{'status':'COMPLETED_REVIEW_ONLY','n_cases':6,'n_review_frames':96,
        'model_inference_calls':0,'scientific_variant_added':False,'cases':receipts,'wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
