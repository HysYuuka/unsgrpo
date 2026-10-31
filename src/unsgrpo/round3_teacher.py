"""Frozen, bounded event-referenced teacher: parsing and target construction."""
import json
import numpy as np
from .core import _no_constant, _unique_object

ROLES = {'background','context','process','key_action','result','uncertain'}
TASK_A = 'Using the shared event reference, judge how important each clip is for understanding the core events of this video.'
TASK_B = 'Using the shared event reference, judge how much key event information a viewer would lose without each clip. This is a judgment, not an actual deletion experiment.'
SCALE = ('Scores: 0=background, transition or unrelated; 1=limited context; 2=relevant process; '
         '3=important step or key event; 4=key result, decisive change or core display. '
         'Static products and results can be important. More motion is not automatically more important. '
         'Use visible evidence only; do not invent unseen actions.')


def strict_json(text):
    return json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_object)


def parse_events(text):
    obj = strict_json(text)
    if not isinstance(obj,dict) or set(obj)!={'events'} or not isinstance(obj['events'],list) or not 1<=len(obj['events'])<=6:
        raise ValueError('Expected 1..6 visible events')
    expected = [f'E{i+1}' for i in range(len(obj['events']))]
    for e,ident in zip(obj['events'],expected):
        if not isinstance(e,dict) or set(e)!={'id','description'} or e['id']!=ident:
            raise ValueError('Event IDs must be unique consecutive E1..E6')
        if not isinstance(e['description'],str) or not e['description'].strip() or len(e['description'])>500:
            raise ValueError('Invalid visible event description')
    return obj


def parse_semantic(text, n, event_ids):
    obj = strict_json(text)
    if not isinstance(obj,dict) or set(obj)!={'scores','event_ids','roles'}:
        raise ValueError('Expected exactly scores/event_ids/roles')
    if any(not isinstance(obj[k],list) or len(obj[k])!=n for k in obj):
        raise ValueError('Semantic arrays must match actual clip count')
    for a,ee,role in zip(obj['scores'],obj['event_ids'],obj['roles']):
        if type(a) is not int or not 0<=a<=4 or role not in ROLES:
            raise ValueError('Illegal semantic score or role')
        if not isinstance(ee,list) or any(type(e) is not str or e not in event_ids for e in ee) or len(set(ee))!=len(ee):
            raise ValueError('Invalid event reference')
        if a>=3 and not ee: raise ValueError('High score needs at least one event reference')
    return obj


def combine(q_excl, first, second, alpha):
    q = np.asarray(q_excl,dtype=float)
    if q.ndim!=1 or not len(q) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError('Invalid exclusion target')
    if alpha not in (.75,.50): raise ValueError('Only the two predeclared coefficients are permitted')
    z = np.zeros(len(q)); u = np.zeros(len(q)); reasons=[]
    for i in range(len(q)):
        if first is None or second is None:
            reasons.append('invalid_semantic_response'); continue
        a,b = first['scores'][i],second['scores'][i]
        u[i]=(a+b)/8.
        if first['roles'][i]=='uncertain' or second['roles'][i]=='uncertain': reasons.append('uncertain_role')
        elif abs(a-b)>1: reasons.append('semantic_disagreement_gt_one')
        else: z[i]=1. if a==b else .75; reasons.append(None)
    q1=(1-alpha*z)*q+alpha*z*u
    c=.25+.75*z
    return q1,c,z,u,reasons


def event_messages():
    return [{'role':'system','content':'Describe visible events only. Return strict JSON, no markdown or explanations.'},
        {'role':'user','content':'<image>\nThis is the existing chronological overview of a complete video. List at most six visible events as a shared reference. Do not use outside titles or categories. Return exactly {"events":[{"id":"E1","description":"short visible description"}]}, with consecutive IDs E1 to E6. Do not infer actions that are not visible.'}]


def semantic_messages(row, ids, events, view, format_version=0):
    if view not in ('A','B') or format_version not in (0,1): raise ValueError('Unregistered prompt')
    system = ('Return strict JSON only, without markdown or reasoning. '+SCALE+
        ' Output exactly three arrays named scores, event_ids, roles, each with one entry per listed clip. '
        'scores are integers 0..4; event_ids entries are arrays of the shared IDs or []; '
        'scores 3 or 4 must cite at least one shared event. roles are background, context, process, key_action, result, uncertain.')
    if format_version:
        system += ' JSON syntax example for two clips: {"scores":[0,3],"event_ids":[[],["E1"]],"roles":["background","key_action"]}. Produce the actual number of entries, not necessarily two.'
    lines=['<image>','The image is the SAME chronological overview of the complete video.',
        'Shared event reference: '+json.dumps(events,ensure_ascii=False),TASK_A if view=='A' else TASK_B,
        f'There are exactly {len(ids)} clips in this scoring packet, listed chronologically. Each separate video below has exactly four cached frames for ONE identified clip, not a new continuous video.']
    for i in ids:
        lo,hi=row['bounds'][i]; lines += [f'Clip {i}, original interval [{lo:.6f}, {hi:.6f}) seconds:', '<video>']
    return [{'role':'system','content':system},{'role':'user','content':'\n'.join(lines)}]
