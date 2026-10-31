import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('archive_round1',Path(__file__).resolve().parents[1]/'scripts/archive_round1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(tmp_path,monkeypatch):
    def saved(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return {'path':str(path),'sha256':m.sha(path)}
    merged=tmp_path/'merged';merged.mkdir();(merged/'model.safetensors').write_bytes(b'explicit small CPU fixture weight')
    common=saved(tmp_path/'common.json',{'status':'COMPLETED','merge_validation_passed':True,
        'acceptance_basis':'verified_bf16_tensor_merge_common_s0','merged_model':str(merged),
        'merged_model_files':{'model.safetensors':m.sha(merged/'model.safetensors')}})
    runs={}
    for method in m.METHODS:
        base=tmp_path/'runs'/method;checkpoint=base/'output'/'v0'/'checkpoint-256';checkpoint.mkdir(parents=True)
        (checkpoint/'adapter_model.safetensors').write_bytes(b'explicit fixture adapter')
        (checkpoint/'optimizer.pt').write_bytes(b'explicit fixture optimizer')
        files={p.name:m.sha(p) for p in checkpoint.iterdir()}
        (base/'training_ledger.jsonl').write_text('{"fixture":true}\n')
        (base/'output'/'v0'/'logging.jsonl').write_text('{"loss":1}\n')
        runs[method]=saved(base/'receipt.json',{'status':'COMPLETED','mode':'formal','gpu_training_verified':True,
            'method_id':method,'endpoint':{'path':str(checkpoint),'files':files},'output_dir':str(base/'output')})
    metadata=tmp_path/'labels_eval_only'/'conversion_receipt.json';ref=saved(metadata,{'status':'COMPLETED','fixture':True})
    source_index=tmp_path/'sources.json';source_index.write_text(json.dumps({'prefix':'fixture','common_s0':common,'runs':runs,
        'metadata':[dict(ref,destination='evaluation/conversion_receipt.json')]}))
    archive_parent=tmp_path/'archives';monkeypatch.setenv('UNSGRPO_ARCHIVE_DIR',str(archive_parent))
    monkeypatch.setattr(m,'storage_check',lambda out,payload:{'peak_extra_bytes':payload+16*1024**2,'fixture_budget_only':True})
    return source_index,archive_parent/'fixture',tmp_path/'archive_receipt.json'


def test_copy_verify_no_delete_and_reject_existing(tmp_path,monkeypatch):
    index,out,receipt=fixture(tmp_path,monkeypatch)
    result=m.archive(index,out,receipt)
    assert result['status']=='COMPLETED' and result['local_sources_deleted'] is False
    assert result['total_bytes']==sum(item['bytes'] for item in result['files'])
    assert m.sha(result['archive_manifest']['path'])==result['archive_manifest']['sha256']
    assert any(item['destination'].endswith('optimizer.pt') for item in result['files'])
    assert any(item['destination'].endswith('logging.jsonl') for item in result['files'])
    for item in result['files']:
        assert Path(item['source']).is_file()
        assert m.sha(item['source'])==m.sha(out/item['destination'])==item['sha256']
    with pytest.raises(ValueError):m.archive(index,out,receipt)


@pytest.mark.parametrize('failure',['source_hash','annotation','destination_escape','budget'])
def test_failures_preserve_failed_receipt_and_sources(tmp_path,monkeypatch,failure):
    index,out,receipt=fixture(tmp_path,monkeypatch);obj=json.loads(index.read_text())
    if failure=='source_hash':obj['metadata'][0]['sha256']='0'*64
    elif failure=='annotation':
        path=tmp_path/'labels.mat';path.write_bytes(b'not copied');obj['metadata']=[{'path':str(path),'sha256':m.sha(path),'destination':'labels.mat'}]
    elif failure=='destination_escape':obj['metadata'][0]['destination']='../escape.json'
    else:
        def reject(*args):raise ValueError('fixture peak budget rejected')
        monkeypatch.setattr(m,'storage_check',reject)
    index.write_text(json.dumps(obj))
    with pytest.raises(ValueError):m.archive(index,out,receipt)
    assert json.loads(receipt.read_text())['status']=='FAILED'
    assert all(Path(ref['path']).is_file() for ref in obj['runs'].values())
    assert not (tmp_path/'escape.json').exists()


def test_storage_check_enforces_archive_group_and_reserve(monkeypatch):
    report={'ok':True,'groups':{'file_archives':{'allocated_bytes':49*m.GIB}},
            'filesystems':{'file':{'free_bytes':100*m.GIB}}}
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=json.dumps(report),stderr=''))
    with pytest.raises(ValueError):m.storage_check(Path('/unused'),2*m.GIB)
    report['groups']['file_archives']['allocated_bytes']=0;report['filesystems']['file']['free_bytes']=21*m.GIB
    with pytest.raises(ValueError):m.storage_check(Path('/unused'),2*m.GIB)
