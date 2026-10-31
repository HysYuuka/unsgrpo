"""Actual ffmpeg/PyAV timestamp tests; generated fixtures are not real-data claims."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest

@pytest.fixture
def prepare_module():
    spec=importlib.util.spec_from_file_location('prepare_visual',Path(__file__).parents[1]/'scripts/prepare_visual.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('duration,filter_expr',[(3.32,None),(.32,None),(2.04,None),(2.4,"select='if(lt(t,1.2),not(mod(n,3)),1)'")])
def test_pts_end_and_short_video(prepare_module,duration,filter_expr):
    from unsgrpo.data import clip_intervals,sample_targets
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);video=root/'fixture.mp4';out=root/'frames';out.mkdir()
        cmd=['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=160x120:rate=25:duration={duration}']
        if filter_expr: cmd+=['-vf',filter_expr,'-vsync','vfr']
        cmd+=['-c:v','libx264','-pix_fmt','yuv420p',str(video)]
        subprocess.run(cmd,check=True)
        actual,_=prepare_module.probe(video);bounds=clip_intervals(actual)
        paths,pts,error=prepare_module.decode(video,out,bounds)
        targets=sample_targets(bounds)
        assert np.asarray(pts).shape==(len(bounds),4)
        signed=np.asarray(pts)-targets
        held={r['sample_index'] for r in error['terminal_frame_holds']}
        assert all(v>=-1e-9 or i in held for i,v in enumerate(signed.reshape(-1)))
        assert np.max(np.abs(signed))<=.12+1e-6
        if duration==2.04:
            assert len(held)==4 and len(bounds)==2
        assert all(Path(p).is_file() for row in paths for p in row)
        native=np.load(out/'frame_bounds.npy')
        assert native[0,0]==0 and native[-1,1]==actual
        assert np.all(native[:,1]>native[:,0])
        assert error['decoded_native_frames']==len(native)
