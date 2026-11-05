import json,numpy as np,hashlib
from collections import Counter
from pathlib import Path
for ixfile in (2,3):
 p=Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[ixfile]);d=json.loads(p.read_bytes()); assert d['complete']
 g={}; ref=None; pairs={}
 for c in d['conditions']:
  block=c['id'].split('_')[0] if ixfile==3 else 'absolute'
  r={x['prompt_id']:x for x in c['rows']}; assert len(r)==len(c['rows'])==60
  prompts={k:x['prompt'] for k,x in r.items()};ref=prompts if ref is None else ref;assert ref==prompts
  k=(block,c['receiver'],c['amount'],c['source'],c['random_seed']);assert k not in g;g[k]=r
  for x in r.values():
   v=x['independent_review'];assert v['outcome'] in ('acknowledgment','instructions','unsupported','other');assert v['a']==v['b'] or v['adjudicated'];assert pairs.setdefault((x['prompt'],x['response']),v)==v
 assert (len(g),len(pairs),sum(v['adjudicated'] for v in pairs.values()))==((26,1148,7) if ixfile==2 else (20,1047,113))
 ix=np.random.default_rng(42).integers(0,60,(20000,60))
 def ci(v):return np.r_[v.mean(),np.quantile(v[ix].mean(1),[.025,.975])].__mul__(100).round(4).tolist()
 out={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'plans':d.get('component_plan_sha256',d.get('plan_sha256')),'conditions':[],'contrasts':[],'bootstrap':{'draws':20000,'seed':42,'unit':'paired prompt'}}
 for k,r in g.items():out['conditions'].append(dict(condition=k,counts=dict(Counter(x['independent_review']['outcome'] for x in r.values())),hit_cap=sum(x['hit_token_limit'] for x in r.values()),input_truncated=sum(x['input_truncated'] for x in r.values())))
 for block,n,a,s,seed in g:
  if n!='original_instruct' or not a:continue
  keys=[(block,n,0,'both',None),(block,n,a,s,seed),(block,'safety_lora',0,'both',None),(block,'safety_lora',a,s,seed)]
  row=dict(block=block,dose=a,source=s,seed=seed)
  for label in ('acknowledgment','unsupported'):
   v=[np.array([int(g[k][i]['independent_review']['outcome']==label) for i in ref]) for k in keys]
   row[label]={'counts':[int(x.sum()) for x in v],'contrast_pp':ci(v[3]-v[2]-v[1]+v[0])}
  out['contrasts'].append(row);print(row)
 print('caps',Counter({(b,n):sum(x['hit_cap'] for x in out['conditions'] if x['condition'][:2]==(b,n)) for b,n,*_ in g}))
 Path('artifacts/reproduced/results/'+('llama_absolute' if ixfile==2 else 'qwen_replication')+'_results.json').write_text(json.dumps(out,indent=2)+'\n')
