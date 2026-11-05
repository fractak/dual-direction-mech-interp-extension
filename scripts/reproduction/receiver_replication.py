import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
path=Path('artifacts/review/receiver_replication_resolved/reviewed_source.json')
d=json.loads(path.read_bytes())
assert d['complete'] is True
groups={}
ref=None
votes={}
for c in d['conditions']:
 r={x['prompt_id']:x for x in c['rows']}
 assert len(r)==len(c['rows'])==60
 prompts={k:x['prompt'] for k,x in r.items()}
 if ref is None: ref=prompts
 assert ref==prompts
 key=(c['receiver'],c['mode'])
 assert key not in groups and c['layer']==20 and c['amount']==(.3 if c['mode']=='relative_l2' else 0)
 groups[key]=r
 for x in r.values():
  vote=x['independent_review']
  assert vote['outcome'] in ('acknowledgment','instructions','unsupported','other')
  if vote['a']!=vote['b']: assert vote['adjudicated']
  pair=(x['prompt'],x['response'])
  assert votes.setdefault(pair,vote)==vote
assert len(votes)==510 and sum(v['adjudicated'] for v in votes.values())==33
names=['original_instruct','safety42','safety43','safety44','alpaca42']
assert set(groups)=={(n,m) for n in names for m in ('observe','relative_l2')}
ix=np.random.default_rng(42).integers(0,60,(20000,60))
def ci(v):
 return [round(float(v.mean()*100),4),*[round(float(x),4) for x in np.quantile(v[ix].mean(1)*100,[.025,.975])]]
out={'reviewed_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'component_plan_sha256':d['component_plan_sha256'],'review':{'unique_pairs':510,'adjudicated':33,'agreement':477/510,'kappa':.9002962085308057},'bootstrap':{'unit':'paired prompt','draws':20000,'seed':42},'conditions':{},'contrasts':{}}
for n in names:
 out['conditions'][n]={}
 for m in ('observe','relative_l2'):
  rows=groups[n,m].values()
  out['conditions'][n][m]={'counts':dict(Counter(x['independent_review']['outcome'] for x in rows)),'hit_token_limit':sum(x['hit_token_limit'] for x in rows),'input_truncated':sum(x['input_truncated'] for x in rows)}
for label in ('acknowledgment','unsupported'):
 v={n:np.array([int(groups[n,'relative_l2'][k]['independent_review']['outcome']==label)-int(groups[n,'observe'][k]['independent_review']['outcome']==label) for k in ref]) for n in names}
 out['contrasts'][label]={'change':{n:ci(v[n]) for n in names},'minus_original':{n:ci(v[n]-v[names[0]]) for n in names[1:]},'minus_alpaca':{n:ci(v[n]-v['alpaca42']) for n in names[1:4]}}
Path('artifacts/reproduced/results/receiver_replication_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['conditions']);print(out['contrasts'])
