import sys; sys.argv=['x']
import importlib.util
spec=importlib.util.spec_from_file_location("h","scripts/humaneval_evalplus.py")
# we only want the label() fn + runner, but the module loads the model. Re-implement label inline instead.
import subprocess,tempfile,os,pickle
RPATH="/tmp/_ep_runner.py"  # already written by a prior run
from evalplus.data import get_human_eval_plus
P=get_human_eval_plus()
def label(entry,canonical,candidate,base,plus,atol):
    with tempfile.NamedTemporaryFile("wb",suffix=".pkl",delete=False) as f:
        pickle.dump({"entry":entry,"canonical":canonical,"candidate":candidate,"base":base,"plus":plus,"atol":atol},f); pk=f.name
    try:
        r=subprocess.run(["python3",RPATH,pk],capture_output=True,text=True,timeout=40)
        out=r.stdout.strip().split(); return (int(out[0]),int(out[1])) if len(out)==2 else (0,0,r.stderr[:200])
    finally: os.unlink(pk)

# Pick has_close_elements: canonical compares sorted adjacent diffs.
k=list(P)[0]; p=P[k]; entry=p['entry_point']; prompt=p['prompt']; canon=prompt+p['canonical_solution']
base=p['base_input']; plus=p['plus_input']; atol=p['atol']
print("entry",entry,"atol",atol)
# 1) canonical vs itself -> must be (1,1)
print("canonical:", label(entry,canon,canon,base,plus,atol))
# 2) subtly wrong: use <= instead of <, OR only check unsorted adjacent (classic HumanEval weak-test FP)
buggy_unsorted = prompt + '''
    for i in range(len(numbers)-1):
        if abs(numbers[i+1]-numbers[i])<threshold:
            return True
    return False
'''
print("buggy(adjacent-only, no sort):", label(entry,canon,buggy_unsorted,base,plus,atol))
# 3) blatantly wrong: always False
wrong = prompt + "\n    return False\n"
print("always-False:", label(entry,canon,wrong,base,plus,atol))
