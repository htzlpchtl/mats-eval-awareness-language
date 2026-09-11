"""Exact experimental flow and reported results; no model inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figures/presentation'
INK = '#233140'
GREY = '#65717C'
C = {'Benchmark Eval':'#B65B23', 'Benchmark Deploy':'#246E92',
     'Casual Eval':'#865CA6', 'Casual Deploy':'#167B73'}


def draw():
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig = plt.figure(figsize=(12.6, 7.4), facecolor='white')
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    def text(x,y,s,size=11,color=INK,ha='left',va='center',**kw):
        ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,**kw)
    def box(x,y,w,h,fill='#F7F9FA',edge='#DCE3E8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.012',
                                  facecolor=fill,edgecolor=edge,lw=1.1))
    def groups(x,y,w,h,title,n,labels):
        box(x,y,w,h)
        text(x+.015,y+h-.030,title,12)
        text(x+.015,y+h-.061,n,9.7,GREY)
        top=y+h-.104
        step=.037 if len(labels)==4 else .045
        for i,label in enumerate(labels):
            yy=top-i*step
            ax.plot([x+.017,x+.017],[yy-.010,yy+.010],lw=3,color=C[label],solid_capstyle='round')
            text(x+.029,yy,label,11.5,C[label])
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.25,color='#84939E'))
    text(.035,.957,'How the two probes were trained and tested',20)
    for x,s in [(.035,'TRAINING DATA'),(.343,'FITTED PROBE'),(.516,'HELD-OUT TEST DATA'),(.81,'RESULT')]:
        text(x,.884,s,10.4,GREY)
    # Probe A is fitted once, then branched to two distinct held-out contrasts.
    groups(.035,.536,.242,.218,'Two groups','420 prompts · 210 per group',
           ['Benchmark Eval','Casual Deploy'])
    box(.323,.583,.151,.130,fill='#EDF3F7',edge='#BCD0DE')
    text(.3985,.666,'Probe A',17,'#246E92',ha='center')
    text(.3985,.618,'Same fitted probe',9.8,GREY,ha='center')
    arrow((.277,.648),(.323,.648))
    ax.plot([.474,.493,.493],[.648,.648,.737],color='#84939E',lw=1.25)
    ax.plot([.493,.493],[.648,.520],color='#84939E',lw=1.25)
    arrow((.493,.737),(.516,.737));arrow((.493,.520),(.516,.520))
    groups(.516,.635,.245,.205,'Original formats','180 prompts · 90 per group',
           ['Benchmark Eval','Casual Deploy'])
    groups(.516,.418,.245,.205,'Formats reversed','180 prompts · 90 per group',
           ['Casual Eval','Benchmark Deploy'])
    for y,auc,acc,reading in [(.737,'1.000','100% accuracy','Perfect ranking'),
                              (.520,'0.000','3.3% accuracy','Reversed ranking')]:
        arrow((.761,y),(.796,y))
        text(.812,y+.046,reading,11)
        text(.812,y+.002,auc,23)
        text(.927,y-.001,'AUROC',9.8,GREY)
        text(.812,y-.040,acc,11,GREY)
    # Probe B is a new classifier, with both formats represented under both labels.
    ax.plot([.035,.975],[.384,.384],color='#E0E6EA',lw=1)
    four=['Benchmark Eval','Casual Eval','Benchmark Deploy','Casual Deploy']
    groups(.035,.070,.242,.275,'All four groups','840 prompts · 210 per group',four)
    box(.323,.143,.151,.130,fill='#EDF3F7',edge='#BCD0DE')
    text(.3985,.227,'Probe B',17,'#246E92',ha='center')
    text(.3985,.178,'New probe fitted',9.8,GREY,ha='center')
    arrow((.277,.208),(.323,.208));arrow((.474,.208),(.516,.208))
    groups(.516,.070,.245,.275,'All four groups','360 prompts · 90 per group',four)
    arrow((.761,.208),(.796,.208))
    text(.812,.254,'Strong classification',11)
    text(.812,.210,'0.997',23)
    text(.927,.207,'AUROC',9.8,GREY)
    text(.812,.168,'96.7% accuracy',11,GREY)
    text(.035,.029,'Test prompts were excluded from fitting and layer selection. Higher scores mean eval for both probes.',10,GREY)
    return fig


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    fig=draw()
    for ext in ['png','svg']:
        fig.savefig(OUT/f'07_probe_training_and_results.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
