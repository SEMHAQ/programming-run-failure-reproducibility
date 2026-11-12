"""Publication figures from saved results, without hard-coded performance values."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent/'runtime_packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE=Path(__file__).parent
OUT=BASE/'manuscript'/'figs'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1,
    'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def read(name): return json.loads((BASE/'analysis'/name).read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
colors={'2019':'#0F4D92','2021':'#B64342'}

fig,ax=plt.subplots(figsize=(7.2,2.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
boxes=[(.01,.56,.27,.35,'Public event logs\n2019 Phanon\n2021 PyCharm'),
       (.365,.56,.27,.35,'Replay edits at runs\nKeep recorded-clean\nflag separate'),
       (.72,.56,.27,.35,'Outcome linkage\nExclude ambiguous\n2021 matches'),
       (.01,.06,.27,.32,'Membership manifests\nIndependent all-runs\nand clean LLM draws'),
       (.365,.06,.27,.32,'Student-disjoint fits\nNested fusion\nHistory ablations'),
       (.72,.06,.27,.32,'Within-assignment AUC\nPaired student\nbootstrap intervals')]
for x,y,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',linewidth=1,
                              edgecolor='#0F4D92',facecolor='#EDF3F8'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
for start,end in [((.285,.735),(.355,.735)),((.64,.735),(.71,.735)),
                  ((.855,.55),(.855,.43)),((.855,.425),(.145,.425)),
                  ((.145,.425),(.145,.39)),((.285,.22),(.355,.22)),((.64,.22),(.71,.22))]:
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#4D4D4D','lw':1})
save(fig,'pipeline_revision')

fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharex=True)
labels=['Process','Syntax-free process','Without outcome history','Last-outcome baseline']
keys=['process','syntax_free','no_outcome_history','last_outcome_baseline']
for ax,cohort in zip(axes,['2019','2021']):
    r=read(f'process_{cohort}.json')['clean']['metrics']
    for i,k in enumerate(keys):
        v=r[k]; ax.errorbar(v['auc'],i,xerr=[[v['auc']-v['lo']],[v['hi']-v['auc']]],
                           fmt='o',color=colors[cohort],capsize=3,markersize=4)
    ax.axvline(.5,color='#767676',ls='--',lw=.8)
    ax.set_title(cohort); ax.set_yticks(range(len(keys)),labels if cohort=='2019' else ['']*len(keys))
    ax.invert_yaxis(); ax.set_xlabel('Macro AUC with 95% interval'); ax.set_xlim(.48,.82)
fig.tight_layout(pad=.6,w_pad=1.2)
save(fig,'history_ablation_revision')

fig,ax=plt.subplots(figsize=(7.2,3.0))
for cohort,offset,marker in [('2019',-.06,'o'),('2021',.06,'s')]:
    r=read(f'llm_{cohort}.json')['clean']['metrics']
    x=np.arange(4)+offset
    y=np.array([r[m]['auc'] for m in ['3B','7B','14B','32B']])
    lo=np.array([r[m]['lo'] for m in ['3B','7B','14B','32B']])
    hi=np.array([r[m]['hi'] for m in ['3B','7B','14B','32B']])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=marker+'-',lw=1.1,capsize=3,
                color=colors[cohort],label=cohort,markersize=4)
ax.axhline(.5,color='#767676',ls='--',lw=.8)
ax.set_xticks(np.arange(4),['3B','7B','14B','32B'])
ax.set_xlabel('Qwen2.5 Instruct model size'); ax.set_ylabel('Macro AUC with 95% interval')
ax.set_ylim(.15,1); ax.legend(loc='lower right',ncol=2)
fig.tight_layout(pad=.7)
save(fig,'llm_uncertainty_revision')

fig,axes=plt.subplots(1,2,figsize=(7.2,3),sharex=True)
for ax,cohort in zip(axes,['2019','2021']):
    d=read(f'llm_{cohort}.json')['clean']['deltas']
    for j,m in enumerate(['7B','14B','32B']):
        for method,off,color,marker in [('stack',-.12,'#B64342','s'),('rankmean',.12,'#0F4D92','o')]:
            v=d[f'{m}_{method}_minus_raw']
            ax.errorbar(v['delta'],j+off,xerr=[[v['delta']-v['lo']],[v['hi']-v['delta']]],
                        fmt=marker,color=color,capsize=3,markersize=4,
                        label={'stack':'Logistic stack','rankmean':'Rank average'}[method] if j==0 else None)
    ax.axvline(0,color='#767676',ls='--',lw=.8)
    ax.set_yticks([0,1,2],['7B','14B','32B']); ax.invert_yaxis(); ax.set_title(cohort)
    ax.set_xlabel('Change from raw LLM AUC'); ax.set_xlim(-.23,.16)
handles,legend_labels=axes[1].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.01),ncol=2,fontsize=8)
fig.tight_layout(pad=.7,w_pad=1.5,rect=(0,.1,1,1))
save(fig,'fusion_revision')
print('Generated four vector figures with 300 dpi PNG previews')
