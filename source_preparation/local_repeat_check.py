"""Small independent repeatability audit; original evaluation caches remain untouched."""
import time,json,urllib.request,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))
import reanalyse as ra
BASE='http://localhost:11434'
PROMPT='```python\n%s\n```\n\nWill this program raise an error when executed? Answer Y for yes, N for no. One character only.'
models={'3B':'qwen2.5:3b','7B':'qwen25-7b-8k:latest',
        '14B':'qwen25-14b-8k:latest','32B':'qwen25-32b-8k:latest'}
out=ra.ROOT/'revision'/'analysis'
def req(path,body):
    r=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=600) as response: return json.load(response)
def score(choice):
    lp=(choice.get('logprobs') or {}).get('content') or []
    if not lp: return None
    import math
    first=lp[0]; py=pn=None
    for c in first.get('top_logprobs',[]):
        t=c.get('token','').strip().upper()
        if t=='Y' and py is None: py=math.exp(c['logprob'])
        if t=='N' and pn is None: pn=math.exp(c['logprob'])
    if py is None and pn is None: return None
    if py is None: return 0.
    if pn is None: return 1.
    return py/(py+pn)
selected=[]
for cohort in ['2019','2021']:
    d=ra.load(cohort)
    ids=list(d[d['primary'] & (d['recorded_clean']==1)].sample(5,random_state=20261112)['EventID'])
    rs={str(r['EventID']):r for r in [json.loads(l) for l in (ra.CORPUS/('runset.jsonl' if cohort=='2019' else 'runset2021.jsonl')).open(encoding='utf-8')]}
    for e in ids: selected.append({'cohort':cohort,'EventID':e,'code':rs[e].get('code') or rs[e].get('_code') or ''})
summary={}
rawpath=out/'repeat_check_responses.jsonl'
with rawpath.open('w',encoding='utf-8') as fh:
    for label,name in models.items():
        print('REPEAT',label,flush=True)
        try:
            info=req('/api/show',{'model':name})
            # Warm up on a separate item, then compare the same 10 prompts in two passes.
            warm={'model':name,'temperature':0,'max_tokens':4000,'logprobs':True,'top_logprobs':20,
                  'messages':[{'role':'user','content':PROMPT%selected[0]['code']}]}
            req('/v1/chat/completions',warm)
            records=[]
            for passno in [1,2]:
                for item in selected:
                    body=dict(warm);body['messages']=[{'role':'user','content':PROMPT%item['code']}]
                    start=time.perf_counter(); response=req('/v1/chat/completions',body); elapsed=time.perf_counter()-start
                    ch=response['choices'][0]
                    row={'model':label,'cohort':item['cohort'],'EventID':item['EventID'],
                         'pass':passno,'latency_s':elapsed,'score':score(ch),'choice':ch}
                    records.append(row);fh.write(json.dumps(row,ensure_ascii=False)+'\n');fh.flush()
            first=records[:10];second=records[10:]
            import numpy as np
            times=np.array([r['latency_s'] for r in records])
            comparable=[(a,b) for a,b in zip(first,second) if a['score'] is not None and b['score'] is not None]
            summary[label]={'n_items':10,'requests':20,'score_comparable':len(comparable),
                'identical_answer':sum(a['choice']['message']['content']==b['choice']['message']['content'] for a,b in zip(first,second)),
                'identical_score':sum(a['score']==b['score'] for a,b in comparable),
                'max_abs_score_difference':max((abs(a['score']-b['score']) for a,b in comparable),default=None),
                'latency_median_s':float(np.median(times)),'latency_p25_s':float(np.quantile(times,.25)),
                'latency_p75_s':float(np.quantile(times,.75)),
                'details':info.get('details'),'parameters':info.get('parameters')}
        except Exception as e: summary[label]={'error':type(e).__name__+': '+str(e)}
        ra.save_json('repeat_check.json',summary)
print('DONE repeat check',flush=True)
