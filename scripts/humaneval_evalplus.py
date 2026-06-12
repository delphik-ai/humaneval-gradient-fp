"""HumanEval verifier false-positive detection, EvalPlus-grounded (objective GT; no rules, no LLM judge).
verifier = HumanEval base tests (few inputs, documented-weak). oracle = EvalPlus plus tests (~999 inputs).
A base-passing solution is a FALSE POSITIVE iff it FAILS plus -> weak verifier accepted a wrong solution.
genuine = passes base AND plus. Save every base-passing solution TEXT + label + gradient.
"""
import json, torch, numpy as np, argparse, re, subprocess, tempfile, os, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from evalplus.data import get_human_eval_plus

ap=argparse.ArgumentParser()
ap.add_argument("--model",default="Qwen/Qwen2.5-Coder-3B-Instruct")
ap.add_argument("--n_problems",type=int,default=60)
ap.add_argument("--n_samples",type=int,default=8)
ap.add_argument("--out",default="results/ep")
args=ap.parse_args()

PROB=get_human_eval_plus()
keys=list(PROB)[:args.n_problems]

tok=AutoTokenizer.from_pretrained(args.model)
model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,device_map="cuda")
model.config.use_cache=False
lcfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["q_proj","v_proj"])
model=get_peft_model(model,lcfg)
gseed=torch.Generator().manual_seed(0)
for n_,p in model.named_parameters():
    if "lora_B" in n_: p.data=(torch.randn(p.shape,generator=gseed)*0.01).to(device=p.device,dtype=p.dtype)
    p.requires_grad_("lora_" in n_)
model.train(False)
lp=[p for n_,p in model.named_parameters() if p.requires_grad]

RUNNER=r'''
import pickle,sys,math,copy
d=pickle.load(open(sys.argv[1],'rb')); entry=d['entry']; atol=d['atol'] or 1e-6
def eq(a,b):
    if isinstance(a,bool) or isinstance(b,bool): return type(a)==type(b) and a==b
    if isinstance(a,float) or isinstance(b,float):
        try:
            if math.isnan(a) and math.isnan(b): return True
        except Exception: pass
        try: return abs(a-b)<=max(atol,1e-6)
        except Exception: return a==b
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    return a==b
nc={}; exec(d['canonical'],nc); canon=nc[entry]
ng={}
try: exec(d['candidate'],ng); cand=ng[entry]
except Exception: print('0 0'); sys.exit()
def passes(inputs):
    for inp in inputs:
        try: e=canon(*copy.deepcopy(inp))
        except Exception: continue
        try: gg=cand(*copy.deepcopy(inp))
        except Exception: return False
        if not eq(gg,e): return False
    return True
bp=passes(d['base']); pp=passes(d['plus']) if bp else False
print(int(bp),int(pp))
'''
RPATH="/tmp/_ep_runner.py"; open(RPATH,"w").write(RUNNER)

def label(entry,canonical,candidate,base,plus,atol):
    with tempfile.NamedTemporaryFile("wb",suffix=".pkl",delete=False) as f:
        pickle.dump({"entry":entry,"canonical":canonical,"candidate":candidate,"base":base,"plus":plus,"atol":atol},f); pk=f.name
    try:
        r=subprocess.run(["python3",RPATH,pk],capture_output=True,text=True,timeout=40)
        out=r.stdout.strip().split()
        bp,pp=(int(out[0]),int(out[1])) if len(out)==2 else (0,0)
    except Exception: bp,pp=0,0
    finally: os.unlink(pk)
    return bp,pp

def gen(prompt,n):
    msgs=[{"role":"user","content":f"Complete this Python function. Return ONLY the complete function (def line + body) in a single python code block.\n\n{prompt}"}]
    p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=tok(p,return_tensors="pt").to("cuda"); outs=[]
    with torch.no_grad():
        o=model.generate(**enc,do_sample=True,temperature=1.0,top_p=0.95,max_new_tokens=400,num_return_sequences=n,pad_token_id=tok.eos_token_id)
    plen=enc.input_ids.shape[1]
    for j in range(n):
        txt=tok.decode(o[j][plen:],skip_special_tokens=True)
        m=re.search(r"```(?:python)?\s*\n(.*?)```",txt,re.S)
        outs.append((p,m.group(1).strip() if m else txt.strip()))
    return outs

def gradvec(prompt,sol):
    ids=tok(prompt+sol+tok.eos_token,return_tensors="pt",truncation=True,max_length=1600).to("cuda")
    plen=tok(prompt,return_tensors="pt").input_ids.shape[1]
    labels=ids.input_ids.clone(); labels[0,:plen]=-100
    model.zero_grad(set_to_none=True)
    model(input_ids=ids.input_ids,attention_mask=ids.attention_mask,labels=labels).loss.backward()
    v=torch.cat([q.grad.detach().flatten() for q in lp]).float()
    return (v/(v.norm()+1e-8)).cpu().numpy()

recs=[]; V=[]; fout=open(args.out+"_passing.jsonl","w")
print("=== generate -> label base/plus (EvalPlus GT) -> keep base-passers -> save text+gradient ===")
for pi,k in enumerate(keys):
    pr=PROB[k]; prompt=pr["prompt"]; entry=pr["entry_point"]; canon=prompt+pr["canonical_solution"]
    base=pr["base_input"]; plus=pr["plus_input"]; atol=pr.get("atol",1e-6)
    nfp=0
    for p,sol in gen(prompt,args.n_samples):
        code = sol if f"def {entry}" in sol else prompt+"\n"+sol
        bp,pp=label(entry,canon,code,base,plus,atol)
        if bp:
            lab="genuine" if pp else "FP"
            rec={"pid":pi,"entry":entry,"idx":len(recs),"label":lab,"prompt":prompt,"solution":sol}
            recs.append(rec); V.append(gradvec(prompt,sol)); fout.write(json.dumps(rec)+"\n"); fout.flush()
            if lab=="FP": nfp+=1
    nb=sum(1 for r in recs if r['pid']==pi)
    print(f"  prob {pi+1}/{len(keys)} {entry}: base-passers={nb} FP={nfp} | cum base={len(recs)} cumFP={sum(1 for r in recs if r['label']=='FP')}",flush=True)
fout.close()
np.savez(args.out+"_grad.npz",V=np.stack(V),pids=np.array([r['pid'] for r in recs]),labels=np.array([r['label'] for r in recs]))
from collections import Counter
print(f"\nDONE. base-passers={len(recs)} | {dict(Counter(r['label'] for r in recs))} -> {args.out}_passing.jsonl + _grad.npz")
