"""Relabel saved base-passers with the OFFICIAL EvalPlus harness (untrusted_check),
then compute gradient sim-to-rest AUC (overall + within-problem). No custom comparator.
"""
import json, numpy as np
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval import untrusted_check

P=get_human_eval_plus(); ks=list(P)
print("computing/loading official groundtruth outputs...")
GT=get_groundtruth(P, get_human_eval_plus_hash(), [])

recs=[json.loads(l) for l in open('results/ep_full_passing.jsonl')]
D=np.load('results/ep_full_grad.npz',allow_pickle=True)
V=D['V']; assert len(recs)==len(V), (len(recs),len(V))
print(f"loaded {len(recs)} base-passers (my labeler), gradient dim {V.shape[1]}")

def official(rec):
    p=P[ks[rec['pid']]]; tid=p['task_id']
    sol=rec['solution']; code = sol if f"def {p['entry_point']}" in sol else p['prompt']+"\n"+sol
    exp=GT[tid]
    b,_=untrusted_check("humaneval",code,p['base_input'],p['entry_point'],
        expected=exp["base"],atol=p["atol"],ref_time=exp["base_time"],fast_check=True)
    if b!="pass": return "base_fail"
    pl,_=untrusted_check("humaneval",code,p['plus_input'],p['entry_point'],
        expected=exp["plus"],atol=p["atol"],ref_time=exp["plus_time"],fast_check=True)
    return "genuine" if pl=="pass" else "FP"

newlab=[]
for i,r in enumerate(recs):
    newlab.append(official(r))
    if (i+1)%100==0: print(f"  relabeled {i+1}/{len(recs)}",flush=True)

from collections import Counter
agree=sum(1 for r,n in zip(recs,newlab) if r['label']==n)
print("\n=== relabel summary (official EvalPlus harness) ===")
print("official:",dict(Counter(newlab)))
print("mine    :",dict(Counter(r['label'] for r in recs)))
print(f"agreement {agree}/{len(recs)}")
print("disagreements:")
for r,n in zip(recs,newlab):
    if r['label']!=n: print(f"  pid {r['pid']} {r['entry']}: mine={r['label']} official={n}")

# keep official base-passers only
keep=[i for i,n in enumerate(newlab) if n!="base_fail"]
lab=np.array([newlab[i] for i in keep]); Vk=V[keep]; pid=np.array([recs[i]['pid'] for i in keep])
y=(lab=="FP").astype(int)
print(f"\nanalysis set: {len(keep)} solutions, FP={y.sum()}, FP problems={len(set(pid[y==1]))}")

S=Vk@Vk.T
n=len(Vk)
sim_rest=(S.sum(1)-S.diagonal())/(n-1)

def auc(score,y):  # P(score_FP < score_genuine) convention -> use -score for "FP lower"
    pos=score[y==1]; neg=score[y==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    r=np.argsort(np.argsort(np.concatenate([pos,neg])))+1
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

print("\n=== gradient signal: sim-to-rest (global) ===")
print(f"mean sim FP={sim_rest[y==1].mean():.4f}  genuine={sim_rest[y==0].mean():.4f}")
print(f"AUC(FP has LOWER sim-to-rest) = {auc(-sim_rest,y):.3f}   [0.5=none]")

# within-problem: z-score sim-to-OWN-problem peers, only problems with both labels
wz=[]; wy=[]
for q in sorted(set(pid)):
    m=pid==q
    if m.sum()<3 or len(set(y[m]))<2: continue
    Sq=Vk[m]@Vk[m].T; nq=m.sum()
    sr=(Sq.sum(1)-Sq.diagonal())/(nq-1)
    z=(sr-sr.mean())/(sr.std()+1e-9)
    wz.append(z); wy.append(y[m])
if wz:
    wz=np.concatenate(wz); wy=np.concatenate(wy)
    print(f"\n=== within-problem (problems with both labels): n={len(wy)}, FP={wy.sum()} ===")
    print(f"AUC(FP lower within-problem sim) = {auc(-wz,wy):.3f}")
else:
    print("\nno problem has both FP and genuine with n>=3")

# also: sim to same-problem genuine only (cleaner reference set)
out={"n":int(len(keep)),"fp":int(y.sum()),
     "auc_global":float(auc(-sim_rest,y)),
     "auc_within":float(auc(-wz,wy)) if len(wz)>0 else None,
     "labels_official":[str(x) for x in lab.tolist()],"pids":[int(x) for x in pid.tolist()]}
json.dump(out,open('results/ep_analysis.json','w'),indent=1)
print("\nsaved results/ep_analysis.json")
