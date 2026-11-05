import json,hashlib,numpy as np
from collections import Counter
from pathlib import Path
o={}
for n,f in [(4,'llama'),(5,'qwen')]:
 p=Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[n]); d=json.loads(p.read_bytes())
 assert d['complete'] is True
 def select(d,old=False):
  cs=[c for c in d['conditions'] if not old or f=='llama' or (c['id'].startswith('relative_') and c['source']!='random')]
  cs=sorted(cs,key=lambda c:(c['receiver'],c['mode']!='observe'))
  assert [(c['receiver'],c['mode']) for c in cs]==[(a,b) for a in ('original_instruct','safety_lora') for b in ('observe','relative_l2')]
  return [{x['prompt_id']:x for x in c['rows']} for c in cs]
 r=select(d); old=select(json.loads(Path({0: 'artifacts/review/length_sensitivity_256_resolved/reviewed_source.json', 1: 'artifacts/review/receiver_replication_resolved/reviewed_source.json', 2: 'artifacts/review/absolute_l2_resolved/reviewed_source.json', 3: 'artifacts/review/qwen_replication_resolved/reviewed_source.json', 4: 'artifacts/review/length_sensitivity_512/llama_resolved/reviewed_source.json', 5: 'artifacts/review/length_sensitivity_512/qwen_resolved/reviewed_source.json', 6: 'artifacts/review/fresh_absolute_resolved/reviewed_source.json', 7: 'artifacts/review/learned_controls_resolved/reviewed_source.json', 8: 'artifacts/review/reextract_resolved/reviewed_source.json', 9: 'artifacts/review/fixed_sign_resolved/reviewed_source.json', 10: 'artifacts/review/phase_crossover_resolved/reviewed_source.json'}[0 if f=='llama' else 3]).read_text()),True)
 ids=list(r[0]); assert len(ids)==60
 for x in r+old: assert {i:t['prompt'] for i,t in x.items()}=={i:t['prompt'] for i,t in r[0].items()}
 votes={}
 for x in r:
  for t in x.values():
   v=t['independent_review'];assert v['outcome'] in ('acknowledgment','instructions','unsupported','other')
   assert v['a']==v['b'] or v['adjudicated'];assert votes.setdefault((t['prompt'],t['response']),v)==v
 assert len(votes)==(240 if f=='llama' else 238) and sum(v['adjudicated'] for v in votes.values())==5
 unc=[i for i in ids if all(not x[i]['hit_token_limit'] for x in r)]
 summary={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'plan_sha256':d['plan_sha256'],'caps':[sum(t['hit_token_limit'] for t in x.values()) for x in r],'input_truncated':[sum(t['input_truncated'] for t in x.values()) for x in r],'n_uncapped':len(unc),'unique_pairs':len(votes),'disagreements':5,'all_counts':[dict(Counter(t['independent_review']['outcome'] for t in x.values())) for x in r]}
 for name,ids0 in [('all',ids),('uncapped_512',unc)]:
  ix=np.random.default_rng(42).integers(0,len(ids0),(20000,len(ids0)))
  def ci(v): return np.round(100*np.r_[v.mean(),np.quantile(v[ix].mean(1),[.025,.975])],4).tolist()
  summary[name]={}
  for label in ('acknowledgment','unsupported'):
   def get(rr):
    v=[np.array([int(x[i]['independent_review']['outcome']==label) for i in ids0]) for x in rr]
    return [int(x.sum()) for x in v],v[3]-v[2]-v[1]+v[0]
   a,v=get(r);b,w=get(old)
   summary[name][label]={'counts_512':a,'contrast_512':ci(v),'counts_256':b,'contrast_256':ci(w),'contrast_difference':ci(v-w)}
 o[f]=summary
 print(f,summary)
Path('artifacts/reproduced/results/length_sensitivity_512_results.json').write_text(json.dumps(o,indent=2)+'\n')
