import subprocess,tempfile,os,pickle,copy
RPATH="/tmp/_ep_runner.py"
from evalplus.data import get_human_eval_plus
P=get_human_eval_plus()
def label(entry,canonical,candidate,base,plus,atol):
    with tempfile.NamedTemporaryFile("wb",suffix=".pkl",delete=False) as f:
        pickle.dump({"entry":entry,"canonical":canonical,"candidate":candidate,"base":base,"plus":plus,"atol":atol},f); pk=f.name
    try:
        r=subprocess.run(["python3",RPATH,pk],capture_output=True,text=True,timeout=40)
        out=r.stdout.strip().split(); return (int(out[0]),int(out[1])) if len(out)==2 else ('ERR',r.stderr[:300])
    finally: os.unlink(pk)
k=list(P)[0]; p=P[k]; entry=p['entry_point']; prompt=p['prompt']; canon=prompt+p['canonical_solution']
base=p['base_input']; plus=p['plus_input']; atol=p['atol']
# compute canonical outputs on the 7 base inputs, build a lookup-by-position cheater that passes base, fails plus
ns={}; exec(canon,ns); cf=ns[entry]
outs=[cf(*copy.deepcopy(b)) for b in base]
cheat = prompt + "\n    _b=%r\n    _o=%r\n    import copy\n    for _i,_x in enumerate(_b):\n        if [numbers,threshold]==_x: return _o[_i]\n    return False\n" % (base,outs)
print("cheater (hardcodes 7 base answers, else False):", label(entry,canon,cheat,base,plus,atol), "<-- expect (1, 0) = a TRUE false positive")
