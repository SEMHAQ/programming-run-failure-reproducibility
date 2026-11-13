"""Update the original recovery comparison using the audited primary population.

No student code is executed. Recovery models are refit within assignment on the
previous-failure subset, preserving the original experiment's training population.
"""
from pathlib import Path
import json
import reanalyse as a
import pandas as pd

def main():
    result={}
    for year in ['2019','2021']:
        path=a.OUT/f'recovery_predictions_{year}.csv'
        if path.exists():
            scored=pd.read_csv(path,dtype={'EventID':str,'SubjectID':str})
        else:
            d=a.load(year)
            d=d[d['primary'] & (d['last_outcome']==1)].copy()
            pieces=[]
            with a.threadpool_limits(limits=4):
                for assignment,s in d.groupby('AssignmentID',sort=True):
                    if len(s)<150 or s.SubjectID.nunique()<5 or s.y.nunique()<2:continue
                    s=s.reset_index(drop=True).copy()
                    print(year,assignment,len(s),flush=True)
                    s['syntax_free']=a.oof(s,a.PURE)
                    s['process']=a.oof(s,a.PROCESS)
                    s['static_syntax_baseline']=1-s.recorded_clean
                    pieces.append(s)
            scored=pd.concat(pieces,ignore_index=True)
        # Export only public identifiers, outcomes and derived held-out scores.
        scored=scored[['EventID','SubjectID','AssignmentID','y','syntax_free',
                       'process','static_syntax_baseline']].copy()
        scored.to_csv(path,index=False)
        metrics,_=a.bootstrap(scored,['syntax_free','process','static_syntax_baseline'])
        cells=[dict(assignment=str(g),n=len(s),failure_rate=float(s.y.mean()))
               for g,s in scored.groupby('AssignmentID')]
        result[year]=dict(n=len(scored),students=int(scored.SubjectID.nunique()),
                          metrics=metrics,per_assignment=cells)
    a.save_json('recovery.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
