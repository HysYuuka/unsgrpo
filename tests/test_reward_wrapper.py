"""Installed Swift CPU reward path, including its actual Arrow image wrapper."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

pytest.importorskip('swift')
from swift.dataset import RowPreprocessor
from swift.rl_core.data import GRPOSample
from swift.rl_core.grpo_algorithm import score_completions
from unsgrpo.core import reward_parts
from unsgrpo.data import build_window_record,clip_intervals,sample_targets

spec=importlib.util.spec_from_file_location('reward_image_wrapper_plugin',Path(__file__).resolve().parents[1]/'scripts/reward_plugin.py')
plugin_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_module)


def make_contract(tmp_path,monkeypatch):
    bounds=clip_intervals(4.)
    row=build_window_record(video_id='cpu_contract',source_group='cpu_contract',split='train',
        duration=4.,bounds=bounds,frame_paths=[[str(tmp_path/f'{i}_{j}.jpg') for j in range(4)] for i in range(2)],
        sampled_pts=sample_targets(bounds),overview_path=str(tmp_path/'overview.jpg'),clip_ids=[0,1],
        pseudo_scores=[.25,.75],task='grpo')
    row.update(dataset_request_id=row['request_id'],pool_instance_id='cpu_pool_0')
    data=tmp_path/'train_grpo.jsonl';data.write_text(json.dumps(row)+'\n')
    monkeypatch.setenv('HL_GRPO_DATA',str(data));monkeypatch.setenv('HL_ROLLOUT_LEDGER',str(tmp_path/'rollouts.jsonl'))
    monkeypatch.setenv('HL_GROUP_SIZE','4');monkeypatch.setenv('HL_RANK_WEIGHT','.8');monkeypatch.setenv('RANK','0')
    plugin=plugin_module.HighlightRecurrenceReward()
    texts=['{"scores":[25,75]}','{"scores":[75,25]}','{"scores":[50,50]}','{"scores":[']
    samples=[]
    for index,text in enumerate(texts):
        actual=copy.deepcopy(row)
        RowPreprocessor._cast_mm_data(actual)
        assert actual['images']==[{'bytes':None,'path':row['images'][0]}]
        assert actual['videos']==row['videos']
        actual['messages'].append({'role':'assistant','content':text})
        sample=GRPOSample.from_row(actual);sample.prompt_id='prompt_0';sample.request_id=f'cpu-uuid-{index}'
        samples.append(sample)
    return plugin,row,texts,samples


def calculate(plugin,samples):
    return score_completions(samples,reward_funcs=[plugin],reward_model_plugins=[None],
                             use_gym_env=False,device=torch.device('cpu'))


def test_actual_swift_wrapper_preserves_reward_and_ledger(tmp_path,monkeypatch):
    plugin,row,texts,samples=make_contract(tmp_path,monkeypatch)
    values=calculate(plugin,samples)
    np.testing.assert_allclose(values[:,0].tolist(),[reward_parts(t,row['pseudo_scores']).total for t in texts],atol=1e-7)
    ledger=[json.loads(x) for x in plugin.ledger.read_text().splitlines()]
    assert len(ledger)==4 and ledger[-1]['reward']['total']==-1
    assert all(r['images_paths_normalized']==[str(Path(row['images'][0]).resolve())] for r in ledger)


@pytest.mark.parametrize('mutation',['path','bytes','extra_key','videos'])
def test_wrapper_cannot_hide_misaligned_content(tmp_path,monkeypatch,mutation):
    plugin,row,texts,samples=make_contract(tmp_path,monkeypatch)
    if mutation=='path':samples[1].images[0]['path']=str(tmp_path/'different_content.jpg')
    elif mutation=='bytes':samples[1].images[0]['bytes']=b'unexpected image bytes'
    elif mutation=='extra_key':samples[1].images[0]['unexpected']=None
    else:samples[1].videos[0][0]=str(tmp_path/'different_video_frame.jpg')
    with pytest.raises(ValueError):calculate(plugin,samples)
    assert not plugin.ledger.exists()
