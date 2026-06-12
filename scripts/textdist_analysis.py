"""Regime analysis + text-distance baseline.

Inputs (in results/): ep_full_passing.jsonl, ep_status.json, ep_final.json, ep_sims.json
  (ep_sims.json holds per-solution gradient similarity values, extracted from ep_full_grad.npz
   by computing, within each problem, every solution's mean cosine similarity to its siblings,
   plus a within-problem z-score. Regenerate via humaneval_evalplus.py + this file's --extract.)

Measures:
  1. text_dist: per solution, mean normalized edit distance (1 - difflib.SequenceMatcher.ratio)
     to same-problem siblings, docstrings/comments stripped. A model-free "how unusual is this text" score.
  2. Spearman correlation between text-difference and gradient outlierness.
  3. FP-detection AUC within-problem, split by regime:
     genuine-majority problems vs FP-majority problems.
"""
import json, re, sys
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict

recs=[json.loads(l) for l in open('results/ep_full_passing.jsonl')]
st=json.load(open('results/ep_status.json'))
fin=json.load(open('results/ep_final.json'))
sims={r['i']:r for r in json.load(open('results/ep_sims.json'))}
keep=[s['i'] for s in st if s['status']!='base_fail']
lab={i:l for i,l in zip(keep,fin['labels'])}

def norm(sol):
    i=sol.find('def ')
    t=sol[i:] if i>=0 else sol
    t=re.sub(r'"""[\s\S]*?"""','',t); t=re.sub(r'#[^\n]*','',t)
    return '\n'.join(l.rstrip() for l in t.splitlines() if l.strip())

bypid=defaultdict(list)
for i in keep: bypid[recs[i]['pid']].append(i)
rows=[]
for pid,idx in bypid.items():
    if len(idx)<3: continue
    texts={i:norm(recs[i]['solution']) for i in idx}
    for a in idx:
        ds=[1-SequenceMatcher(None,texts[a],texts[b]).ratio() for b in idx if b!=a]
        rows.append({"i":a,"pid":pid,"entry":recs[a]['entry'],"label":lab[a],
                     "text_dist":sum(ds)/len(ds),"z_within":sims[a]['z_within']})
byp=defaultdict(list)
for r in rows: byp[r['pid']].append(r)
for pid,rs in byp.items():
    td=np.array([r['text_dist'] for r in rs]); m,s=td.mean(),td.std()+1e-9
    for r in rs: r['z_text']=float((r['text_dist']-m)/s)

import scipy.stats as ss
Z=np.array([r['z_within'] for r in rows]); T=np.array([r['z_text'] for r in rows])
rho,p=ss.spearmanr(T,-Z)
print(f"n={len(rows)} solutions / {len(byp)} problems")
print(f"Spearman(text-difference, gradient-outlierness) = {rho:.3f} (p={p:.2e})")

def auc(score,y):
    pos=score[y==1]; neg=score[y==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    r=np.argsort(np.argsort(np.concatenate([pos,neg])))+1
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

MIXED={8,48,49,55,86,89,97,154,158}
fp_n=defaultdict(int); tot_n=defaultdict(int)
for r in rows:
    if r['pid'] in MIXED:
        tot_n[r['pid']]+=1
        if r['label']=='FP': fp_n[r['pid']]+=1
minority=[p for p in MIXED if fp_n[p]/tot_n[p]<0.5]
majority=[p for p in MIXED if fp_n[p]/tot_n[p]>=0.5]
for name,pids in [("genuine-majority",minority),("FP-majority",majority),("pooled",list(MIXED))]:
    rs=[r for r in rows if r['pid'] in pids]
    y=np.array([1 if r['label']=='FP' else 0 for r in rs])
    Zm=np.array([r['z_within'] for r in rs]); Tm=np.array([r['z_text'] for r in rs])
    print(f"{name:>17}: n={len(rs):>3} FP={y.sum():>2}  gradient AUC={auc(-Zm,y):.3f}  text-dist AUC={auc(Tm,y):.3f}")

# --- cohesion: a problem's mean pairwise gradient similarity (no labels needed) ---
coh={}
for pid,rs in byp.items():
    coh[pid]=float(np.mean([sims[r['i']]['sim_within'] for r in rs]))
order=sorted(coh,key=lambda p:coh[p])
loose_rank={p:k+1 for k,p in enumerate(order)}
print("\ncohesion of the 5 genuine-majority problems (loosest-rank of",len(coh),"problems):")
for p,name in [(48,'is_palindrome'),(8,'sum_product'),(55,'fib'),(158,'find_max'),(89,'encrypt')]:
    print(f"  {name:<15} cohesion={coh[p]:.3f}  loosest-rank {loose_rank[p]}")
