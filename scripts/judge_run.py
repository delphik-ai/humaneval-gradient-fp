"""Blind LLM judge: per problem, score each solution 1-5 on how different its
algorithmic approach is from the other solutions in the set. No labels shown."""
import json, os, sys, re
from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION","2024-10-21"))
DEP=os.environ["AZURE_OPENAI_DEPLOYMENT"]
tasks=json.load(open(sys.argv[1]))
out=[]
for t in tasks:
    listing="\n\n".join(f"### Solution {s['key']}\n```python\n{s['code']}\n```" for s in t['solutions'])
    prompt=f"""You are shown {len(t['solutions'])} Python solutions to the same programming problem (function `{t['entry']}`).

For EACH solution, rate on a 1-5 scale how DIFFERENT its algorithmic approach and code structure are from the other solutions in this set:
1 = essentially the same approach as most of the others
3 = same general approach but notably different structure or idioms
5 = a completely different approach from the rest

Judge only the approach/structure. Do NOT judge correctness, style quality, or efficiency.

{listing}

Answer with ONLY a JSON object mapping each solution letter to its score, e.g. {{"A": 1, "B": 4}}."""
    r=client.chat.completions.create(model=DEP,messages=[{"role":"user","content":prompt}])
    txt=r.choices[0].message.content
    m=re.search(r'\{[^{}]*\}',txt)
    scores=json.loads(m.group(0))
    out.append({"pid":t['pid'],"entry":t['entry'],
                "scores":[{"i":s['i'],"key":s['key'],"judge":int(scores[s['key']])} for s in t['solutions']]})
    print(t['entry'],scores,flush=True)
json.dump(out,open('judge_scores.json','w'),indent=1)
print("saved judge_scores.json")
