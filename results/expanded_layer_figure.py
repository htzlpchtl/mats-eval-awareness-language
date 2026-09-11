"""Standalone expanded-scale Probe B layer curve from saved training CV."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def draw():
    d = pd.read_csv(ROOT/'results/metrics/probe_b_layer_cv.csv')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
        'text.color':'#233140','axes.labelcolor':'#233140',
        'xtick.color':'#65717C','ytick.color':'#65717C',
        'axes.edgecolor':'#CDD3D8','axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none'}):
        fig, ax = plt.subplots(figsize=(6.4,3.5),facecolor='white')
        ax.plot(d.layer_number,d.mean_cv_auroc,color='#246E92',lw=1.8)
        ax.fill_between(d.layer_number,d.min_cv_auroc,d.max_cv_auroc,
                        color='#246E92',alpha=.10,linewidth=0)
        ax.set(xlim=(1,32),ylim=(.98,1.001),xlabel='Decoder block',
               ylabel='Mean CV AUROC')
        ax.set_title('Expanded scale',fontsize=13,pad=13)
        ax.set_xticks([1,8,16,24,32]); ax.set_yticks([.98,.99,1.0])
        ax.grid(alpha=.15); ax.set_axisbelow(True)
        fig.subplots_adjust(left=.15,right=.965,bottom=.19,top=.85)
    return fig

if __name__=='__main__':
    out=ROOT/'results/figures/presentation'
    out.mkdir(parents=True,exist_ok=True)
    fig=draw()
    for ext in ['png','svg']:
        fig.savefig(out/f'08_probe_b_layers_expanded.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
