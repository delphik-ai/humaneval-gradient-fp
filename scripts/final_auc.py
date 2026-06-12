"""Final: stable wrong-output FPs only (double-checked official fails), recompute gradient AUCs, dump FP samples."""
import json, numpy as np
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval import untrusted_check
P=get_human_eval_plus(); ks=list(P)
GT=get_groundtruth(P, get_human_eval_plus_hash(), [])
recs=[json.loads(l) for l in open('results/ep_full_passing.jsonl')]
st=json.load(open('results/ep_status.json'))
D=np.load('results/ep_full_grad.npz',allow_pickle=True); V=D['V']

# re-check the 52 plus_fail rows once more; FP = fails BOTH times (stable wrong-output)
stable={}
for s in st:
    if s['status']!='plus_fail': continue
    r=recs[s['i']]; p=P[ks[r['pid']]]
    sol=r['solution']; code = sol if f"def {p['entry_point']}" in sol else p['prompt']+"\n"+sol
    exp=GT[p['task_id']]
    pl,_=untrusted_check("humaneval",code,p['plus_input'],p['entry_point'],expected=exp["plus"],atol=p["atol"],ref_time=exp["plus_time"],fast_check=False)
    stable[s['i']]= (pl!='pass')
print(f"recheck of 52 plus_fail: stable fails={sum(stable.values())}, flipped to pass={sum(1 for v in stable.values() if not v)}")

lab=[]
for s in st:
    if s['status']=='base_fail': lab.append('drop')
    elif s['status']=='plus_fail' and stable.get(s['i'],False): lab.append('FP')
    else: lab.append('genuine')   # plus_pass + unstable fails -> conservative genuine
keep=[i for i,l in enumerate(lab) if l!='drop']
y=np.array([1 if lab[i]=='FP' else 0 for i in keep]); Vk=V[keep]
pid=np.array([recs[i]['pid'] for i in keep])
print(f"final analysis set n={len(keep)}, FP={y.sum()} across {len(set(pid[y==1]))} problems")

S=Vk@Vk.T; n=len(Vk)
sim_rest=(S.sum(1)-S.diagonal())/(n-1)
def auc(score,yy):
    pos=score[yy==1]; neg=score[yy==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    r=np.argsort(np.argsort(np.concatenate([pos,neg])))+1
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def auc_se(a,n1,n2):  # Hanley-McNeil
    q1=a/(2-a); q2=2*a*a/(1+a)
    return np.sqrt((a*(1-a)+(n1-1)*(q1-a*a)+(n2-1)*(q2-a*a))/(n1*n2))
a_g=auc(-sim_rest,y); se=auc_se(a_g,int(y.sum()),int((1-y).sum()))
print(f"\nGLOBAL sim-to-rest: mean FP={sim_rest[y==1].mean():.4f} genuine={sim_rest[y==0].mean():.4f}")
print(f"AUC={a_g:.3f} (SE~{se:.3f}, ~95% CI {a_g-1.96*se:.3f}-{a_g+1.96*se:.3f})")

wz=[];wy=[]
for q in sorted(set(pid)):
    m=pid==q
    if m.sum()<3 or len(set(y[m]))<2: continue
    Sq=Vk[m]@Vk[m].T; nq=int(m.sum())
    sr=(Sq.sum(1)-Sq.diagonal())/(nq-1)
    z=(sr-sr.mean())/(sr.std()+1e-9)
    wz.append(z);wy.append(y[m])
wz=np.concatenate(wz);wy=np.concatenate(wy)
a_w=auc(-wz,wy); se_w=auc_se(a_w,int(wy.sum()),int((1-wy).sum()))
print(f"\nWITHIN-PROBLEM (mixed problems only): n={len(wy)} FP={wy.sum()}")
print(f"AUC={a_w:.3f} (SE~{se_w:.3f}, ~95% CI {a_w-1.96*se_w:.3f}-{a_w+1.96*se_w:.3f})")

# per-mixed-problem detail: does the FP rank lowest in its problem?
print("\nper-problem rank of FPs (1=lowest sim):")
for q in sorted(set(pid)):
    m=pid==q
    if m.sum()<3 or len(set(y[m]))<2: continue
    Sq=Vk[m]@Vk[m].T; nq=int(m.sum())
    sr=(Sq.sum(1)-Sq.diagonal())/(nq-1)
    order=np.argsort(sr); ranks={int(o):j+1 for j,o in enumerate(order)}
    fpr=[ranks[int(j)] for j in np.where(y[m]==1)[0]]
    print(f"  pid {q} {recs[keep[int(np.where(m)[0][0])]]['entry']}: n={nq} FP_ranks={fpr}")

# dump 6 diverse FP samples for human read
fps=[(i,recs[i]) for i in keep if lab[i]=='FP']
seen=set(); dump=[]
for i,r in fps:
    if r['pid'] in seen: continue
    seen.add(r['pid']); dump.append({"pid":r['pid'],"entry":r['entry'],"solution":r['solution']})
    if len(dump)>=8: break
json.dump(dump,open('results/ep_fp_samples.json','w'),indent=1)
json.dump({"n":int(len(keep)),"fp":int(y.sum()),"auc_global":float(a_g),"auc_within":float(a_w),
  "labels":[lab[i] for i in keep],"pids":[int(x) for x in pid.tolist()]},open('results/ep_final.json','w'))
print("\nsaved ep_final.json + ep_fp_samples.json")
