import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
paths=[Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[i]) for i in (6,9,10)]
ds=[json.loads(p.read_bytes()) for p in paths]
g={}; ref=None
names=['original_instruct','safety42']; modes=['baseline','AA','FF','AF','FA']
pairs={}
for j,d in enumerate(ds):
 assert d['complete']
 for c in d['conditions']:
  n=c['receiver'];n='safety42' if n=='safety_lora' else n
  if j==0:
   if n not in names or c['source'] not in ('both','evidence'):continue
   mode='baseline' if c['source']=='both' else 'AA'
  elif j==1:
   if c['mode']!='absolute_l2_subtract':continue
   mode='FF'
  else:
   mode={'adaptive_prefill_fixed_decode':'AF','fixed_prefill_adaptive_decode':'FA'}[c['mode']]
   assert c['source']=='evidence'
  assert c['layer']==20 and c['amount']==(0 if mode=='baseline' else 4)
  r={x['prompt_id']:x for x in c['rows']};assert len(r)==len(c['rows'])==120 and (n,mode) not in g
  q={k:(x['prompt'],x['domain']) for k,x in r.items()};ref=q if ref is None else ref;assert ref==q
  g[n,mode]=r
  for x in r.values():
   v=x['independent_review'];assert v['outcome'] in ('acknowledgment','instructions','unsupported','other') and (v['a']==v['b'] or v['adjudicated'])
   if j==2:assert pairs.setdefault((x['prompt'],x['response']),v)==v
assert len(pairs)==465 and sum(v['adjudicated'] for v in pairs.values())==74 and set(g)=={(n,m) for n in names for m in modes}
families=sorted({v[1] for v in ref.values()});assert len(families)==30
out={'source_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'plan_sha256':ds[2]['plan_sha256'],'bootstrap':{'seed':42,'draws':20000,'unit':'paired family'},'conditions':{},'analyses':{}}
for k,r in g.items():
 out['conditions']['/'.join(k)]={'counts':dict(Counter(x['independent_review']['outcome'] for x in r.values())),'cap':sum(x['hit_token_limit'] for x in r.values()),'input_truncated':sum(x['input_truncated'] for x in r.values())}
print(out['conditions'])
for subset in ('all','uncapped'):
 ids=[k for k in ref if subset=='all' or all(not r[k]['hit_token_limit'] for r in g.values())]
 fams=sorted({ref[k][1] for k in ids})
 m=np.array([[ref[k][1]==f for k in ids] for f in fams],dtype=int)
 ix=np.random.default_rng(42).integers(0,len(fams),(20000,len(fams)))
 def ci(v):
  b=100*(m@v)[ix].sum(1)/m.sum(1)[ix].sum(1)
  return np.round(np.r_[100*v.mean(),np.quantile(b,[.025,.975])],4).tolist()
 for label in ('acknowledgment','unsupported'):
  for vote in ('outcome','a','b'):
   z={k:np.array([int(r[i]['independent_review'][vote]==label) for i in ids]) for k,r in g.items()}
   rec={m:z[names[1],m]-z[names[1],'baseline']-z[names[0],m]+z[names[0],'baseline'] for m in modes[1:]}
   aa,ff,af,fa=(rec[k] for k in ('AA','FF','AF','FA'))
   rec.update(prefill=((aa-fa)+(af-ff))/2,decode=((aa-af)+(fa-ff))/2,
              interaction=aa-af-fa+ff)
   out['analyses']['/'.join((subset,label,vote))]={'n':len(ids),'contrasts':{k:ci(v) for k,v in rec.items()}}
   if subset=='all' or vote=='outcome':print(subset,label,vote,out['analyses']['/'.join((subset,label,vote))])
Path('artifacts/reproduced/results/phase_crossover_results.json').write_text(json.dumps(out,indent=2)+'\n')
