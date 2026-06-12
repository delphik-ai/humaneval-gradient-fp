"""Task-level gradient comparison: average each problem's solution gradients into one
task vector, normalize, compute each task's mean cosine similarity to all other tasks."""
import json, numpy as np
recs=[json.loads(l) for l in open('/home/jongwon/ep_full_passing.jsonl')]
st=json.load(open('/home/jongwon/ep_status.json'))
D=np.load('/home/jongwon/ep_full_grad.npz',allow_pickle=True); V=D['V']
keep=[s['i'] for s in st if s['status']!='base_fail']
from collections import defaultdict
bypid=defaultdict(list)
for k,i in enumerate(keep): bypid[recs[i]['pid']].append(k)
pids=sorted(bypid)
M=[]
meta=[]
Vk=V[keep]
for p in pids:
    m=Vk[bypid[p]].mean(0)
    m=m/ (np.linalg.norm(m)+1e-12)
    M.append(m); meta.append({"pid":int(p),"entry":recs[[i for i in keep if recs[i]['pid']==p][0]]['entry'],"n":len(bypid[p])})
M=np.stack(M).astype(np.float32)
S=M@M.T; n=len(M)
sim=(S.sum(1)-S.diagonal())/(n-1)
for k in range(n): meta[k]["task_sim_to_rest"]=float(sim[k])
json.dump(meta,open('/home/jongwon/ep_task_sims.json','w'))
print("tasks:",n,"-> /home/jongwon/ep_task_sims.json")
