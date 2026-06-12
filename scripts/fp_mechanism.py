"""For each stable-FP problem, find the first officially-failing plus input and classify: wrong output vs slow/crash."""
import json, copy, numpy as np, signal
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
P=get_human_eval_plus(); ks=list(P)
GT=get_groundtruth(P, get_human_eval_plus_hash(), [])
recs=[json.loads(l) for l in open('results/ep_full_passing.jsonl')]
fin=json.load(open('results/ep_final.json'))
st=json.load(open('results/ep_status.json'))
fp_idx=[s['i'] for s in st if s['status']=='plus_fail']
seen=set()
class TO(Exception): pass
def alarm(*a): raise TO()
signal.signal(signal.SIGALRM, alarm)
for i in fp_idx:
    r=recs[i]
    if r['pid'] in seen: continue
    seen.add(r['pid'])
    p=P[ks[r['pid']]]; exp=GT[p['task_id']]['plus']
    sol=r['solution']; code = sol if f"def {p['entry_point']}" in sol else p['prompt']+"\n"+sol
    ns={}
    try: exec(code,ns)
    except Exception as e: print(f"pid {r['pid']} {r['entry']}: exec error {e}"); continue
    f=ns[p['entry_point']]
    verdict=None
    for j,inp in enumerate(p['plus_input']):
        e=exp[j]
        signal.alarm(5)
        try: g=f(*copy.deepcopy(inp))
        except TO: verdict=("SLOW(>5s)",inp,e,None); signal.alarm(0); break
        except Exception as ex: verdict=(f"CRASH({type(ex).__name__})",inp,e,None); signal.alarm(0); break
        signal.alarm(0)
        ok = (g==e)
        if not ok and isinstance(e,float):
            try: ok = abs(g-e)<=1e-6
            except Exception: pass
        if not ok: verdict=("WRONG_OUTPUT",inp,e,g); break
    if verdict:
        kind,inp,e,g=verdict
        print(f"pid {r['pid']} {r['entry']}: {kind}")
        print(f"    input={repr(inp)[:110]}")
        if kind=="WRONG_OUTPUT": print(f"    expected={repr(e)[:80]}  got={repr(g)[:80]}")
    else:
        print(f"pid {r['pid']} {r['entry']}: no fail found in plain re-run (likely time-limit-only under official harness)")
