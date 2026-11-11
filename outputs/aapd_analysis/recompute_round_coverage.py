import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from aapd_models import load_minilm
from pipeline_metrics import maximum_matching
from pipeline_data import read_truth
root=Path(__file__).resolve().parents[2]
p=root/'outputs/benchmark/AAPD/aligned-v1'
spaces=[(p/'discovery/init_label_space.txt').read_text().splitlines()]
spaces += [json.loads((p/f'improvement/iteration_{i}.json').read_text())['cumulative_labels'] for i in range(10)]
config={'separator':',','strip_prefix':True}
v=sorted({s for f in ['train_label_full.txt','test_label_full.txt'] for row in read_truth(root/'data/AAPD'/f,config) for s in row})
names=sorted(set(v)|{s for space in spaces for s in space})
model=load_minilm('cpu')
vectors=dict(zip(names,model.encode(names,show_progress_bar=False)))
cache=json.loads((p/'evaluation_judgments.json').read_text())
judge='Qwen/Qwen2.5-1.5B-Instruct'
missing=set()
def match(a,b):
    score=float(vectors[a]@vectors[b])
    if score>=.75:return True
    if score<.5:return False
    key=json.dumps([judge,a,b])
    if key not in cache:
        missing.add((a,b));return False
    return cache[key]['match']
results=[]
for i,space in enumerate(spaces):
    edges={a:[b for b in space if match(a,b)] for a in v}
    pairs=maximum_matching(edges)
    results.append({'iteration':i,'matched':len(pairs),'coverage':len(pairs)/len(v),'labels':len(space)})
    print(results[-1],flush=True)
out={'results':results,'missing_judgments':sorted(missing),'note':'Original uncorrected truth parsing; intermediate cumulative label spaces; cached semantic decisions only.'}
(root/'outputs/aapd_analysis/round_coverage.json').write_text(json.dumps(out,indent=2))
print('MISSING',len(missing),flush=True)
