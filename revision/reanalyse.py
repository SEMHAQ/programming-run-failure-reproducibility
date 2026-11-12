"""Deterministic revision analysis; never calls a language model or executes student code.

All fitted predictions use student-disjoint folds. Stacks use inner student folds
inside each outer training set. Bootstrap intervals resample students, paired
across scores, conditional on the stored out-of-fold predictions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import collections
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime_packages'))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'corpus'
OUT = ROOT / 'revision' / 'analysis'
OUT.mkdir(exist_ok=True)
PROCESS = ['edits_since_last_run', 'secs_since_last_run', 'edits_since_clean',
           'secs_since_last_edit', 'blk_run_idx', 'blk_prior_errs', 'is_first_run', 'has_other_sec']
PURE = [c for c in PROCESS if c != 'edits_since_clean']
NOHIST = [c for c in PURE if c != 'blk_prior_errs']
SHAPE = ['buf_len', 'n_lines', 'has_other_sec', 'other_sec_compiles']
FEATURES = {'process': PROCESS, 'syntax_free': PURE, 'no_outcome_history': NOHIST,
            'process_shape': list(dict.fromkeys(PROCESS + SHAPE))}
MODELS = {'3B': 'qwen2.5-3b', '7B': 'qwen25-7b-8k',
          '14B': 'qwen25-14b-8k', '32B': 'qwen25-32b-8k'}
SEED = 20261112
REFIT = False

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load(cohort):
    prepared=ROOT/'data'/f'analysis_input_{cohort}.csv'
    if prepared.exists():
        d=pd.read_csv(prepared,dtype={'EventID':str,'SubjectID':str,'AssignmentID':str})
        for c in ['primary','ambiguous_join']:
            d[c]=d[c].astype(bool)
        return d
    d = pd.read_csv(CORPUS / ('runset.csv' if cohort == '2019' else 'runset2021.csv'),
                    dtype=str, keep_default_na=False)
    if cohort == '2019':
        d = d[d['X-RunUserTerminated'] != 'True'].copy()
        d['y'] = (d['X-RunHasError'] == 'True').astype(int)
        d['ts'] = pd.to_numeric(d['ClientTimestamp'])
        d['ambiguous_join'] = False
    else:
        # Ambiguity is assessed among all exact snapshots, before outcome filtering.
        exact = d['match'] == 'exact'
        ambiguous_keys = set(d.loc[exact, 'RunKey'][d.loc[exact, 'RunKey'].duplicated(keep=False)])
        d = d[exact & d['outcome'].isin(['ok', 'err'])].copy()
        d['y'] = (d['outcome'] == 'err').astype(int)
        d['ts'] = pd.to_numeric(d['ts'])
        d['ambiguous_join'] = d['RunKey'].isin(ambiguous_keys)
    for c in set(PROCESS + SHAPE + ['n_other_sec']) - {'is_first_run', 'has_other_sec', 'blk_prior_errs'}:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for c in ['edits_since_last_run', 'secs_since_last_run', 'secs_since_last_edit', 'other_sec_compiles']:
        d.loc[d[c] < 0, c] = np.nan
    d['has_other_sec'] = (d['n_other_sec'] > 0).astype(int)
    d['is_first_run'] = (d['blk_run_idx'] == 0).astype(int)
    # Compute history before restricting to syntax-clean or unambiguous events.
    # Retain the validated 2019 online counter; 2021 history uses prior labelled events only.
    d['_eid'] = pd.to_numeric(d['EventID'])
    d = d.sort_values(['SubjectID', 'AssignmentID', 'ts', '_eid'], kind='stable').reset_index(drop=True)
    hist = d.groupby(['SubjectID', 'AssignmentID'], sort=False)['y']
    d['last_outcome'] = hist.shift(1)
    if 'blk_prior_errs' not in d:
        d['blk_prior_errs'] = hist.transform(lambda s: s.shift(1).fillna(0).cumsum())
    else:
        d['blk_prior_errs'] = pd.to_numeric(d['blk_prior_errs'])
    d['recorded_clean'] = (d['X-Compilable'] == '1').astype(int)
    manifest = pd.read_csv(ROOT/'revision'/'audit'/f'event_manifest_{cohort}.csv', dtype={'EventID':str})
    d = d.merge(manifest[['EventID','reconstructed_parses','intended_all','intended_clean',
                          'duplicate_excluded','common_llm']], on='EventID', validate='one_to_one')
    d['primary'] = ~d['ambiguous_join']
    return d

def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.08, max_leaf_nodes=31,
        max_depth=None, min_samples_leaf=20, l2_regularization=0, max_bins=255,
        early_stopping=False, random_state=0)

def fit_predict(train, test, cols):
    m = model().fit(train[cols].to_numpy(float), train['y'].to_numpy())
    return m.predict_proba(test[cols].to_numpy(float))[:,1]

def oof(d, cols):
    out = np.full(len(d), np.nan)
    splits = GroupKFold(n_splits=min(5, d['SubjectID'].nunique()))
    for tr, te in splits.split(d, d['y'], d['SubjectID']):
        assert not set(d.iloc[tr]['SubjectID']) & set(d.iloc[te]['SubjectID'])
        out[te] = fit_predict(d.iloc[tr], d.iloc[te], cols)
    assert np.isfinite(out).all()
    return out

def auc_fast(y, s, weights=None):
    # Stable weighted Mann-Whitney statistic, including half credit for ties.
    order = np.argsort(s, kind='stable')
    yy = y[order]; ss = s[order]
    w = np.ones(len(y)) if weights is None else weights[order]
    pos = w * yy; neg = w * (1-yy)
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    gp = np.add.reduceat(pos, starts); gn = np.add.reduceat(neg, starts)
    denom = gp.sum() * gn.sum()
    if denom == 0: return np.nan
    return float(np.sum(gp * (np.cumsum(gn) - .5*gn)) / denom)

def bootstrap(d, cols, n_boot=2000, macro=True):
    y = d['y'].to_numpy(int)
    students, gi = np.unique(d['SubjectID'], return_inverse=True)
    assignments = [np.flatnonzero(d['AssignmentID'].to_numpy()==a)
                   for a in sorted(d['AssignmentID'].unique())]
    assignments = [i for i in assignments if len(np.unique(y[i]))==2]
    if not macro: assignments = [np.arange(len(d))]
    scores = {c:d[c].to_numpy(float) for c in cols}
    point = {c:float(np.mean([auc_fast(y[i],s[i]) for i in assignments])) for c,s in scores.items()}
    rng = np.random.default_rng(SEED)
    # Pre-sort each assignment once so bootstrap only recomputes weighted pair counts.
    prepared = {}
    for c,s in scores.items():
        prepared[c] = []
        for idx in assignments:
            order = idx[np.argsort(s[idx],kind='stable')]
            starts = np.r_[0,np.flatnonzero(np.diff(s[order]))+1]
            prepared[c].append((y[order],gi[order],starts))
    samples = {c:[] for c in cols}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0,len(students),len(students)), minlength=len(students))
        for c in cols:
            vals=[]
            for yy,gg,starts in prepared[c]:
                w=counts[gg]; p=np.add.reduceat(w*yy,starts); n=np.add.reduceat(w*(1-yy),starts)
                denom=p.sum()*n.sum()
                if denom==0: break
                vals.append(float(np.sum(p*(np.cumsum(n)-.5*n))/denom))
            if len(vals)==len(assignments): samples[c].append((b,float(np.mean(vals))))
    result = {}
    for c in cols:
        v=np.array([v for _,v in samples[c]])
        result[c]={'auc':point[c], 'lo':float(np.quantile(v,.025)),
                   'hi':float(np.quantile(v,.975)), 'bootstrap_valid':len(v)}
    return result, {c:dict(samples[c]) for c in cols}

def delta_boot(samples, a, b, point):
    keys=sorted(set(samples[a]) & set(samples[b]))
    v=np.array([samples[a][k]-samples[b][k] for k in keys])
    return {'delta':point[a]['auc']-point[b]['auc'], 'lo':float(np.quantile(v,.025)),
            'hi':float(np.quantile(v,.975)), 'bootstrap_valid':len(v)}

def analyse_process(cohort, d):
    all_summary={}
    for arm in ['all','clean']:
        s=d[d['primary'] & ((d['recorded_clean']==1) if arm=='clean' else True)].copy()
        fname=OUT/f'process_predictions_{cohort}_{arm}.csv'
        if fname.exists() and not REFIT:
            scored=pd.read_csv(fname,dtype={'EventID':str,'SubjectID':str})
        else:
            blocks=[]
            for assignment, sub in s.groupby('AssignmentID',sort=True):
                sub=sub.reset_index(drop=True).copy()
                if len(sub)<200 or sub['SubjectID'].nunique()<5 or sub['y'].nunique()<2: continue
                print('PROCESS',cohort,arm,assignment,len(sub),flush=True)
                for label,cols in FEATURES.items(): sub[label]=oof(sub,cols)
                sub['last_outcome_baseline']=sub['last_outcome'].fillna(.5)
                sub['static_syntax_baseline']=1-sub['recorded_clean']
                blocks.append(sub)
            scored=pd.concat(blocks,ignore_index=True)
            scored.to_csv(fname,index=False)
        cols=list(FEATURES)+['last_outcome_baseline','static_syntax_baseline']
        point,samples=bootstrap(scored,cols)
        per=[]
        for a,sub in scored.groupby('AssignmentID',sort=True):
            q,_=bootstrap(sub,cols)
            per.append({'assignment':a,'n':len(sub),'students':int(sub['SubjectID'].nunique()),
                        'failure_rate':float(sub['y'].mean()),'metrics':q})
        all_summary[arm]={'n':len(scored),'students':int(scored['SubjectID'].nunique()),
             'failure_rate':float(scored['y'].mean()),'metrics':point,'per_assignment':per,
             'deltas':{'shape_minus_process':delta_boot(samples,'process_shape','process',point),
                       'nohistory_minus_syntaxfree':delta_boot(samples,'no_outcome_history','syntax_free',point)}}
        save_json(f'process_{cohort}.json',all_summary)
    # Transfer evaluates common feature semantics; no target-cohort refitting.
    return all_summary

def cache_scores(cohort,d):
    frames=[]
    for label,suffix in MODELS.items():
        fn=('llm_cache_v2_' if cohort=='2019' else 'llm_cache_2021_')+suffix+'.jsonl'
        rr=[json.loads(l) for l in (CORPUS/fn).open(encoding='utf-8') if l.strip()]
        by=collections.defaultdict(list)
        for r in rr: by[str(r['EventID'])].append(r)
        values={e:next((r for r in records if r.get('score') is not None),{}) for e,records in by.items()}
        s=d.copy()
        s[label]=[values.get(e,{}).get('score',np.nan) for e in s['EventID']]
        s[f'{label}_mode']=[('token_probability_retained' if values.get(e,{}).get('lp0',{}).get('token','').strip().upper()
                          in ('Y','N') else 'source_not_retained') for e in s['EventID']]
        frames.append(s[['EventID',label,f'{label}_mode']])
    out=d.copy()
    for s in frames: out=out.merge(s,on='EventID',validate='one_to_one')
    return out[out['common_llm']==1].copy()

def nested_stack(d, score):
    cols=list(dict.fromkeys(PURE+SHAPE))
    out=np.full(len(d),np.nan); llm_only=np.full(len(d),np.nan); proc=np.full(len(d),np.nan)
    ranks=np.full(len(d),np.nan)
    for tr,te in GroupKFold(n_splits=5).split(d,d['y'],d['SubjectID']):
        train=d.iloc[tr].reset_index(drop=True); test=d.iloc[te]
        inner=oof(train,cols)
        held=fit_predict(train,test,cols)
        x=np.column_stack([inner,train[score]])
        meta=LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=1000)
        meta.fit(x,train['y'])
        out[te]=meta.predict_proba(np.column_stack([held,test[score]]))[:,1]
        single=LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=1000)
        single.fit(train[[score]],train['y'])
        llm_only[te]=single.predict_proba(test[[score]])[:,1]
        proc[te]=held
        # Empirical CDFs estimated on outer training data only.
        def cdf(reference, values):
            reference=np.sort(reference)
            return (np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right'))/(2*len(reference))
        ranks[te]=.5*cdf(inner,held)+.5*cdf(train[score].to_numpy(),test[score].to_numpy())
    return out,llm_only,proc,ranks

def analyse_llm(cohort,d):
    s=cache_scores(cohort,d)
    results={}
    for arm in ['all','clean']:
        # Use the documented independent sampling arm; never trust the overwritten _arm field.
        subset=s[s['primary'] & (s['intended_all' if arm=='all' else 'intended_clean']==1)].copy()
        subset=subset.reset_index(drop=True)
        fname=OUT/f'llm_predictions_{cohort}_{arm}.csv'
        if fname.exists() and not REFIT: subset=pd.read_csv(fname,dtype={'EventID':str,'SubjectID':str})
        else:
            # Static parse status is a code-reading baseline; syntax-clean membership remains recorded status.
            subset['static_parse']=1-subset['reconstructed_parses']
            for m in MODELS:
                print('STACK',cohort,arm,m,len(subset),flush=True)
                st,only,pr,rank=nested_stack(subset,m)
                subset[f'{m}_stack']=st; subset[f'{m}_logistic_only']=only
                subset[f'{m}_rankmean']=rank; subset['classical']=pr
            subset.to_csv(fname,index=False)
        # Scoring provenance must be read from the original cache, not inferred from score values.
        for m in MODELS:
            mode=s.set_index('EventID')[f'{m}_mode']
            subset[f'{m}_mode']=subset['EventID'].map(mode)
        keep=[a for a,g in subset.groupby('AssignmentID')
              if g['SubjectID'].nunique()>=5 and g['y'].nunique()==2]
        evalset=subset[subset['AssignmentID'].isin(keep)].copy()
        cols=list(MODELS)+[f'{m}_{v}' for m in MODELS for v in ['stack','logistic_only','rankmean']]+['classical','static_parse']
        point,samples=bootstrap(evalset,cols)
        deltas={}
        for m in MODELS:
            deltas[m+'_stack_minus_raw']=delta_boot(samples,f'{m}_stack',m,point)
            deltas[m+'_stack_minus_logistic']=delta_boot(samples,f'{m}_stack',f'{m}_logistic_only',point)
            deltas[m+'_rankmean_minus_raw']=delta_boot(samples,f'{m}_rankmean',m,point)
        deltas['14B_minus_7B']=delta_boot(samples,'14B','7B',point)
        deltas['32B_minus_14B']=delta_boot(samples,'32B','14B',point)
        per=[]
        for a,g in subset.groupby('AssignmentID',sort=True):
            per.append({'assignment':a,'n':len(g),'students':int(g['SubjectID'].nunique()),
                'failure_rate':float(g['y'].mean()),'included_in_macro':a in keep,
                'auc':{m:auc_fast(g['y'].to_numpy(),g[m].to_numpy()) for m in MODELS}
                      if g['y'].nunique()==2 else {}})
        results[arm]={'sample_n':len(subset),'evaluated_n':len(evalset),
            'students':int(subset['SubjectID'].nunique()),'assignments':keep,
            'failure_rate':float(subset['y'].mean()),'recorded_clean_n':int(subset['recorded_clean'].sum()),
            'metrics':point,'deltas':deltas,'per_assignment':per,
            'score_modes':{m:dict(collections.Counter(subset[f'{m}_mode'])) for m in MODELS}}
        save_json(f'llm_{cohort}.json',results)
    # Report calibration on independent all-runs arm; avoid enrichment by oversampling clean runs.
    subset=s[s['primary'] & (s['intended_all']==1)].copy()
    calibration={}
    for m in MODELS:
        p=subset[m].to_numpy(float); y=subset['y'].to_numpy(int)
        edges=np.unique(np.quantile(p,np.linspace(0,1,11)))
        ece=0.; bins=[]
        for i in range(len(edges)-1):
            mask=(p>=edges[i]) & ((p<=edges[i+1]) if i==len(edges)-2 else (p<edges[i+1]))
            if mask.any():
                ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
                bins.append({'n':int(mask.sum()),'pred':float(p[mask].mean()),'obs':float(y[mask].mean())})
        calibration[m]={'n':len(y),'mean_prediction':float(p.mean()),'observed':float(y.mean()),
            'ece':float(ece),'brier':float(np.mean((p-y)**2)),'saturated':float((p>=.95).mean()),'bins':bins}
    save_json(f'calibration_{cohort}.json',calibration)
    # Ambiguous-key and historical sample sensitivities, no fitted model reuse.
    sensitivity={}
    for tag,mask in {'original_all':s['intended_all']==1,'original_clean':s['intended_clean']==1,
                     'strict_clean':s['primary'] & (s['intended_clean']==1) & (s['reconstructed_parses']==1)}.items():
        z=s[mask].copy()
        keep=[a for a,g in z.groupby('AssignmentID') if g['SubjectID'].nunique()>=5 and g['y'].nunique()==2]
        z=z[z['AssignmentID'].isin(keep)]
        met,_=bootstrap(z,list(MODELS))
        sensitivity[tag]={'n':len(z),'assignments':keep,'metrics':met}
    # Historical summaries excluded assignment cells with fewer than 50 events.
    for arm in ['all','clean']:
        z=s[s['primary'] & (s['intended_all' if arm=='all' else 'intended_clean']==1)].copy()
        keep=[a for a,g in z.groupby('AssignmentID') if len(g)>=50 and g['SubjectID'].nunique()>=5 and g['y'].nunique()==2]
        z=z[z['AssignmentID'].isin(keep)]
        met,_=bootstrap(z,list(MODELS))
        sensitivity['historical_min50_'+arm]={'n':len(z),'assignments':keep,'metrics':met}
    save_json(f'sensitivity_{cohort}.json',sensitivity)
    return results

def transfer(d19,d21):
    out={}
    for source,target,a,b in [('2019','2021',d19,d21),('2021','2019',d21,d19)]:
        tr=a[a['primary'] & (a['recorded_clean']==1)].copy()
        te=b[b['primary'] & (b['recorded_clean']==1)].copy()
        for name,cols in [('syntax_free',PURE),('no_outcome_history',NOHIST)]:
            te[name]=fit_predict(tr,te,cols)
        met,_=bootstrap(te,['syntax_free','no_outcome_history'])
        out[f'{source}_to_{target}']={'train_n':len(tr),'test_n':len(te),'metrics':met}
        te[['EventID','SubjectID','AssignmentID','y','syntax_free','no_outcome_history']].to_csv(
            OUT/f'transfer_{source}_to_{target}.csv',index=False)
    save_json('transfer.json',out)

def main():
    global REFIT
    parser=argparse.ArgumentParser(); parser.add_argument('--part',choices=['process','llm','transfer','all'],default='all')
    parser.add_argument('--refit',action='store_true',help='Retrain predictions instead of using released prediction files')
    args=parser.parse_args()
    REFIT=args.refit
    with threadpool_limits(limits=4):
        datasets={c:load(c) for c in ['2019','2021']}
        for cohort,d in datasets.items():
            if args.part in ['process','all']:
                d[['EventID','SubjectID','AssignmentID','y','primary','ambiguous_join','recorded_clean',
                   'reconstructed_parses','intended_all','intended_clean','duplicate_excluded','common_llm']].to_csv(
                       OUT/f'analysis_membership_{cohort}.csv',index=False)
            print('COHORT',cohort,'primary',int(d['primary'].sum()),'clean',int((d['primary'] & (d['recorded_clean']==1)).sum()),flush=True)
            if args.part in ['process','all']: analyse_process(cohort,d)
            if args.part in ['llm','all']: analyse_llm(cohort,d)
        if args.part in ['transfer','all']: transfer(datasets['2019'],datasets['2021'])
    save_json('runtime.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
        'sklearn':__import__('sklearn').__version__,'scipy':__import__('scipy').__version__,
        'seed':SEED,'bootstrap_replicates':2000})
    print('DONE',args.part,flush=True)

if __name__=='__main__': main()
