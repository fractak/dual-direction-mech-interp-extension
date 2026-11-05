import json,hashlib,numpy as np
from collections import Counter
from pathlib import Path
p=Path('artifacts/review/fresh_absolute_resolved/reviewed_source.json')
d=json.loads(p.read_bytes());assert d['complete'] is True
g={};ref=None;votes={}
for c in d['conditions']:
 k=(c['receiver'],c['source']);assert k not in g and c['layer']==20
 assert (c['mode'],c['amount'])==(('observe',0) if c['source']=='both' else ('absolute_l2',4))
 r={x['prompt_id']:x for x in c['rows']};assert len(r)==len(c['rows'])==120
 g[k]=r
 q={i:(x['prompt'],x['domain']) for i,x in r.items()}
 if ref is None:ref=q
 assert q==ref
 for x in r.values():
  v=x['independent_review'];assert all(v[k] in ('acknowledgment','instructions','unsupported','other') for k in ('outcome','a','b'));assert v['a']==v['b'] or v['adjudicated'];assert votes.setdefault((x['prompt'],x['response']),v)==v
assert len(votes)==1514 and sum(v['adjudicated'] for v in votes.values())==328
fam=sorted({x[1] for x in ref.values()}); assert len(fam)==30 and Counter(x[1] for x in ref.values())==dict.fromkeys(fam,4)
ns=['original_instruct','safety42','safety43','safety44'];ss=['both','safety_lora','evidence','format'];assert set(g)=={(n,s) for n in ns for s in ss}
unc=[i for i in ref if all(not r[i]['hit_token_limit'] for r in g.values())]
out={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'plans':d['component_plan_sha256'],'uncapped_n':len(unc),'unique_pairs':1514,'disagreements':328,'agreement':1186/1514,'kappa':.6736893151570666,'bootstrap':{'unit':'prompt family','seed':42,'draws':20000},'conditions':{},'analyses':{}}
for k,r in g.items():
 out['conditions']['/'.join(k)]={'counts':dict(Counter(x['independent_review']['outcome'] for x in r.values())),'cap':sum(x['hit_token_limit'] for x in r.values()),'truncated_input':sum(x['input_truncated'] for x in r.values())}
for subset,ids in [('all',list(ref)),('uncapped',unc)]:
 if not ids:continue
 families=sorted({ref[i][1] for i in ids})
 membership=np.array([[ref[i][1]==f for i in ids] for f in families],dtype=int)
 boot=np.random.default_rng(42).integers(0,len(families),(20000,len(families)))
 def ci(v):
  bootmean=(membership@v)[boot].sum(1)/membership.sum(1)[boot].sum(1)
  return np.round(100*np.r_[v.mean(),np.quantile(bootmean,[.025,.975])],4).tolist()
 for label in ('acknowledgment','unsupported'):
  for vote in ('outcome','a','b'):
   z={k:np.array([int(r[i]['independent_review'][vote]==label) for i in ids]) for k,r in g.items()}
   rec={s:np.stack([z[n,s]-z[n,'both']-z[ns[0],s]+z[ns[0],'both'] for n in ns[1:]]) for s in ss[1:]}
   rec['evidence_minus_format']=rec['evidence']-rec['format']
   result={s:{'pooled':ci(v.mean(0)),**{n:ci(row) for n,row in zip(ns[1:],v)}} for s,v in rec.items()}
   out['analyses']['/'.join((subset,label,vote))]=result
   print(subset,label,vote,{s:r['pooled'] for s,r in result.items()},result['evidence_minus_format'])
print(out['conditions']);print('uncapped',len(unc))
Path('artifacts/reproduced/results/fresh_absolute_results.json').write_text(json.dumps(out,indent=2)+'\n')
