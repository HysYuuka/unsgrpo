"""Version-scoped type fix for Swift4.6 rollout video frame-list requests.

The processor already accepts List[List[str]] (real SFT/input audit verified).
Rollout's dacite conversion incorrectly narrows videos to List[str]. Extend only
that annotation; preserve request values and every other type check.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import List, Union, get_type_hints

def apply():
    from dacite import from_dict
    from dacite.cache import clear_cache
    from dacite.exceptions import WrongTypeError
    from swift.infer_engine import InferRequest
    from swift.infer_engine.protocol import RolloutInferRequest
    if version('ms-swift')!='4.6.0': raise RuntimeError('Frame-list compatibility fix is scoped to audited Swift4.6.0')
    fixture={'messages':[{'role':'user','content':'<video>'}], 'videos':[['/audit/f0.jpg','/audit/f1.jpg']], 'uuid':'compat-test'}
    old=get_type_hints(RolloutInferRequest)['videos']
    try:
        request=from_dict(RolloutInferRequest,fixture)
        action='ALREADY_SUPPORTED'
    except WrongTypeError as e:
        if e.field_path!='videos' or old!=List[str]: raise
        # Scope the correction to the rollout subclass, without weakening checks.
        RolloutInferRequest.__annotations__['videos']=List[Union[str,List[str]]]
        clear_cache()
        request=from_dict(RolloutInferRequest,fixture)
        action='APPLIED_NESTED_STRING_FRAME_LIST_ANNOTATION'
    if request.videos!=fixture['videos'] or request.messages!=fixture['messages'] or request.uuid!='compat-test':
        raise RuntimeError('Compatibility fix changed request contents')
    # Scalar files remain supported; bad frame element types remain rejected.
    if from_dict(RolloutInferRequest,dict(fixture,videos=['/audit/video.mp4'])).videos!=['/audit/video.mp4']:
        raise RuntimeError('Scalar video path behavior changed')
    try:
        from_dict(RolloutInferRequest,dict(fixture,videos=[[3]]))
    except WrongTypeError:
        pass
    else:
        raise RuntimeError('Invalid nonstring media frames bypassed validation')
    source=Path(inspect.getsourcefile(InferRequest))
    report={'status':'COMPLETED','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'framework_version':version('ms-swift'),'dacite_version':version('dacite'),
            'action':action,'before_annotation':str(old),'after_annotation':str(get_type_hints(RolloutInferRequest)['videos']),
            'installed_source_path':str(source),'installed_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'request_contents_unchanged':True,'invalid_media_types_still_rejected':True,
            'model_math_or_sampling_modified':False}
    if os.environ.get('HL_COMPAT_RECEIPT'):
        path=Path(os.environ['HL_COMPAT_RECEIPT']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    return report

COMPAT_RECEIPT=apply()
