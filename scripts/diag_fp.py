"""Split official-FPs into wrong-output vs timeout/slow, and inspect disagreement cases."""
import json, numpy as np
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval import untrusted_check
P=get_human_eval_plus(); ks=list(P)
GT=get_groundtruth(P, get_human_eval_plus_hash(), [])
recs=[json.loads(l) for l in open('results/ep_full_passing.jsonl')]
ana=json.load(open('results/ep_analysis.json'))
lab=ana['labels_official']  # for kept rows; row 'bf' base_fail was dropped -> rebuild mapping
# recompute official status WITH raw status string for every saved rec (fast_check=False for details)
out=[]
for i,r in enumerate(recs):
    p=P[ks[r['pid']]]
    sol=r['solution']; code = sol if f"def {p['entry_point']}" in sol else p['prompt']+"\n"+sol
    exp=GT[p['task_id']]
    b,_=untrusted_check("humaneval",code,p['base_input'],p['entry_point'],expected=exp["base"],atol=p["atol"],ref_time=exp["base_time"],fast_check=True)
    if b!="pass": out.append((i,r['pid'],r['entry'],'base_'+b)); continue
    pl,det=untrusted_check("humaneval",code,p['plus_input'],p['entry_point'],expected=exp["plus"],atol=p["atol"],ref_time=exp["plus_time"],fast_check=False)
    out.append((i,r['pid'],r['entry'],'plus_'+pl))
    if (i+1)%200==0: print(f"{i+1}/{len(recs)}",flush=True)
from collections import Counter
print("status:",dict(Counter(s for _,_,_,s in out)))
print("\nFP problems by status:")
agg={}
for _,pid,e,s in out:
    if s.startswith('plus_') and s!='plus_pass': agg.setdefault((pid,e),Counter())[s]+=1
for (pid,e),c in sorted(agg.items()): print(f"  pid {pid} {e}: {dict(c)}")
json.dump([{"i":i,"pid":pid,"entry":e,"status":s} for i,pid,e,s in out],open('results/ep_status.json','w'))
print("saved results/ep_status.json")
