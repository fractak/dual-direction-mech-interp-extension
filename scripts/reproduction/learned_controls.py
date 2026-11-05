import json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
paths=[Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[i]) for i in (6,7)]
ds=[json.loads(p.read_bytes()) for p in paths]
g={};ref=None;reviews=[]
names=['original_instruct','safety42','safety43','safety44']
sources=['both','safety_lora','evidence','format','field_order','batch_code']
for d,ss,count,nd in zip(ds,(sources[:4],sources[4:]),(1514,830),(328,55)):
 assert d['complete'] is True
 pairs={}
 for c in d['conditions']:
  k=c['receiver'],c['source'];assert k not in g and k[0] in names and k[1] in ss and c['layer']==20 and c['amount']==(0 if k[1]=='both' else 4)
  r={x['prompt_id']:x for x in c['rows']};assert len(r)==len(c['rows'])==120
  p={i:(x['prompt'],x['domain']) for i,x in r.items()}
  if ref is None: ref=p
  assert ref==p
  g[k]=r
  for x in r.values():
   v=x['independent_review']; assert v['outcome'] in ('acknowledgment','instructions','unsupported','other')
   assert v['a']==v['b'] or v['adjudicated']
   assert pairs.setdefault((x['prompt'],x['response']),v)==v
 assert len(pairs)==count and sum(v['adjudicated'] for v in pairs.values())==nd
 reviews.append(pairs)
assert set(g)=={(n,s) for n in names for s in sources}
overlap=set(reviews[0])&set(reviews[1])
out={'files':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'plans':[d['component_plan_sha256'] for d in ds],'same_pairs_across_batches':len(overlap),'cross_batch_label_conflicts':sum(reviews[0][k]['outcome']!=reviews[1][k]['outcome'] for k in overlap),'bootstrap':{'seed':42,'draws':20000,'unit':'paired prompt family'},'conditions':{},'analyses':{}}
for k,r in g.items():
 out['conditions']['/'.join(k)]={'counts':dict(Counter(x['independent_review']['outcome'] for x in r.values())),'cap':sum(x['hit_token_limit'] for x in r.values())}
print('overlap and conflicts',len(overlap),out['cross_batch_label_conflicts'])
print({k:v for k,v in out['conditions'].items() if k.split('/')[1] in sources[4:]})
for subset in ('all','uncapped'):
 ids=[i for i in ref if subset=='all' or all(not g[n,s][i]['hit_token_limit'] for n in names for s in ('both','evidence',*sources[4:]))]
 families=sorted({ref[i][1] for i in ids});m=np.array([[ref[i][1]==f for i in ids] for f in families],dtype=int)
 assert len(families)==30
 ix=np.random.default_rng(42).integers(0,len(families),(20000,len(families)))
 def ci(v):
  b=100*(m@v)[ix].sum(1)/m.sum(1)[ix].sum(1)
  return {'pp':round(100*float(v.mean()),4),'ci95':np.quantile(b,[.025,.975]).round(4).tolist(),'ci97_5':np.quantile(b,[.0125,.9875]).round(4).tolist()}
 for label in ('acknowledgment','unsupported'):
  for vote in ('outcome','a','b','harmonized'):
   def value(x):
    v=x['independent_review']
    if vote=='harmonized': return reviews[0].get((x['prompt'],x['response']),v)['outcome']
    return v[vote]
   z={k:np.array([int(value(r[i])==label) for i in ids]) for k,r in g.items()}
   rec={s:np.stack([z[n,s]-z[n,'both']-z[names[0],s]+z[names[0],'both'] for n in names[1:]]) for s in ('evidence',*sources[4:])}
   for s in sources[4:]:rec['evidence_minus_'+s]=rec['evidence']-rec[s]
   stats={k:{'pooled':ci(v.mean(0)),**{n:ci(w) for n,w in zip(names[1:],v)}} for k,v in rec.items()}
   out['analyses']['/'.join((subset,label,vote))]={'n':len(ids),'contrasts':stats}
   if (subset=='all' and label=='acknowledgment') or vote=='outcome':print(subset,len(ids),label,vote,{k:v['pooled'] for k,v in stats.items()})
Path('artifacts/reproduced/results/learned_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
