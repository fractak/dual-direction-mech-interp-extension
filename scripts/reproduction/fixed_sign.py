import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
paths=[Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[i]) for i in (6,9)]
ds=[json.loads(p.read_bytes()) for p in paths]
g={}; ref=None
names=['original_instruct','safety42']; modes=['baseline','adaptive','add','subtract']
pairs={}
for j,d in enumerate(ds):
 assert d['complete']
 for c in d['conditions']:
  n=c['receiver'];n='safety42' if n=='safety_lora' else n
  if j==0:
   if n not in names or c['source'] not in ('both','evidence'):continue
   mode='baseline' if c['source']=='both' else 'adaptive'
  else:
   mode={'absolute_l2_add':'add','absolute_l2_subtract':'subtract'}[c['mode']]
   assert c['source']=='evidence'
  assert c['layer']==20 and c['amount']==(0 if mode=='baseline' else 4)
  r={x['prompt_id']:x for x in c['rows']};assert len(r)==len(c['rows'])==120 and (n,mode) not in g
  q={k:(x['prompt'],x['domain']) for k,x in r.items()};ref=q if ref is None else ref;assert ref==q
  g[n,mode]=r
  for x in r.values():
   v=x['independent_review'];assert v['outcome'] in ('acknowledgment','instructions','unsupported','other') and (v['a']==v['b'] or v['adjudicated'])
   if j:assert pairs.setdefault((x['prompt'],x['response']),v)==v
assert len(pairs)==480 and sum(v['adjudicated'] for v in pairs.values())==30 and set(g)=={(n,m) for n in names for m in modes}
families=sorted({v[1] for v in ref.values()});assert len(families)==30
signs={}
for n in names:
 for k in ref:
  vals=[g[n,m][k]['telemetry']['evidence'][0]['signed_projection']['last'] for m in ('add','subtract')]
  assert np.isclose(*vals,atol=1e-6)
  signs[n,k]=vals[0]<0
out={'source_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'plan_sha256':ds[1]['plan_sha256'],'prefill_opposite_sign_prompts':sum(signs[names[0],k]!=signs[names[1],k] for k in ref),'negative_projection_counts':{n:sum(signs[n,k] for k in ref) for n in names},'bootstrap':{'seed':42,'draws':20000,'unit':'paired family'},'conditions':{},'analyses':{}}
out['prefill_has_negative'] = {n:sum(g[n,'add'][k]['telemetry']['evidence'][0]['signed_projection']['min'] < 0 for k in ref) for n in names}
print(out)
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
   rec['add_minus_subtract']=rec['add']-rec['subtract']
   rec['subtract_minus_adaptive']=rec['subtract']-rec['adaptive']
   out['analyses']['/'.join((subset,label,vote))]={'n':len(ids),'contrasts':{k:ci(v) for k,v in rec.items()}}
   if subset=='all' or vote=='outcome':print(subset,label,vote,out['analyses']['/'.join((subset,label,vote))])
Path('artifacts/reproduced/results/fixed_sign_results.json').write_text(json.dumps(out,indent=2)+'\n')
