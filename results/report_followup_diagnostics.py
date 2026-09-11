"""Post-hoc report diagnostics from frozen text and predictions; no refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/metrics/report_followup'
OUT.mkdir(exist_ok=True)
order=['bench_eval','bench_deploy','casual_eval','casual_deploy']
names=['Benchmark eval','Benchmark deploy','Casual eval','Casual deploy']
colours=['#B65B23','#246E92','#865CA6','#167B73']
v1=pd.read_json(ROOT/'data/selected/english_selected.jsonl',lines=True)
v2=pd.read_json(ROOT/'data/selected/english_selected_v2.jsonl',lines=True)
p=pd.read_csv(ROOT/'results/predictions/probe_b_english.csv')
m=p.merge(v2[['stable_id','model_facing_text']],on='stable_id',validate='one_to_one')
m['characters']=m.model_facing_text.str.len()
m['words']=m.model_facing_text.str.split().str.len()
m['contains_digit']=m.model_facing_text.str.contains(r'\d')
m['spaced_punctuation']=m.model_facing_text.str.contains(r'\s+[.!?](?:\s|$)')
m['lowercase_initial']=m.model_facing_text.str.match(r'^[a-z]')
m['error']=m.predicted_class!=m.context_label
m.to_csv(OUT/'english_prompt_characteristics.csv',index=False)
groups=m.groupby(['quadrant','error']).agg(n=('stable_id','size'),median_chars=('characters','median'),median_words=('words','median'),contains_digit=('contains_digit','sum'),spaced_punctuation=('spaced_punctuation','sum'),lowercase_initial=('lowercase_initial','sum')).reset_index()
groups.to_csv(OUT/'english_bucket_summary.csv',index=False)
links=pd.read_csv(ROOT/'results/figures/presentation/06_shared_source_links.csv')
ids=set()
for x in links.itertuples():
    if x.rewritten_split!=x.original_split:
        ids.add(x.rewritten_stable_id if x.rewritten_split=='test' else x.original_stable_id)
clean=m[~m.stable_id.isin(ids)]
stats={'note':'Post-hoc descriptive checks; unchanged fitted probe, not a fresh independent test.',
 'excluded_related_test_rows':len(ids),'remaining_n':len(clean),
 'remaining_auroc':roc_auc_score(clean.context_label,clean.decision_score),
 'remaining_accuracy':accuracy_score(clean.context_label,clean.predicted_class),
 'remaining_errors':int(clean.error.sum())}
(OUT/'summary.json').write_text(json.dumps(stats,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#233140','axes.labelcolor':'#233140','xtick.color':'#65717C','ytick.color':'#65717C','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CDD3D8'})
fig,axs=plt.subplots(1,2,figsize=(8,3.45),sharex=True,sharey=True)
for ax,data,title in zip(axs,[v1,v2],['Initial sample (v1)','After extra matching (v2)']):
    for q,name,c in zip(order,names,colours):
        values=np.sort(data[(data.split=='train')&(data.quadrant==q)].model_facing_text.str.len())
        ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name,color=c,lw=1.7)
    ax.set(title=title,xlim=(0,1200),ylim=(0,1.02),xlabel='Prompt length (characters)')
    ax.set_xticks([0,400,800,1200]);ax.grid(alpha=.15)
axs[0].set_ylabel('Fraction at or below this length')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=9)
fig.subplots_adjust(left=.10,right=.96,top=.89,bottom=.29,wspace=.13)
fig.savefig(OUT/'length_distributions.png',dpi=240,facecolor='white')
plt.close(fig)
print(groups.to_string(index=False));print(json.dumps(stats))
