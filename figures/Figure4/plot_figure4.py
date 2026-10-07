from pathlib import Path

import hashlib, json, re, platform

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator

WIDTH_MM=136

HEIGHT_MM=163

TRIOS='TRIOS'

COMPARATOR='EMAX|EC50|TERTILES'

SEPS=['weak','moderate','strong','ALL']

STYLES={'weak':('#0072B2','o','--'),'moderate':('#D55E00','s','-.'),
        'strong':('#009E73','^',':'),'ALL':('#000000','D','-')}

FIELDS=['ARI','balanced_accuracy','macro_f1','extreme_error']

TITLES=['Adjusted Rand\nindex','Balanced\naccuracy','Macro-F1','Extreme L↔H\nerror (%) ↓']

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

inputs={}

checks=[]

mapping=[]

def style_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7,length=2.7,width=.65,pad=2,colors='black')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis='y',color='#dddddd',linewidth=.4,zorder=0)


def draw(a,b,c):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'axes.edgecolor':'black','axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42,
        'legend.fontsize':7,'legend.frameon':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),facecolor='white')
    left,right=.092,.99
    # Fixed physical-size layout; no tight bounding-box scaling at export.
    gsA=fig.add_gridspec(1,4,left=left,right=right,bottom=.717,top=.870,wspace=.45)
    gsB=fig.add_gridspec(1,2,left=left,right=right,bottom=.428,top=.580,wspace=.25)
    gsC=fig.add_gridspec(1,4,left=left,right=right,bottom=.060,top=.270,wspace=.43)
    fig.text(.012,.970,'A',fontsize=11,fontweight='bold')
    fig.text(.06,.970,'Pharmacodynamic domain extent',fontsize=9.2,fontweight='bold')
    fig.text(.012,.648,'B',fontsize=11,fontweight='bold')
    fig.text(.06,.648,'Finite response resolution',fontsize=9.2,fontweight='bold')
    fig.text(.06,.624,'D = 50 μM; reference K = 7; no refitting',fontsize=7.5)
    fig.text(.71,.624,'Highlighted K=5',fontsize=7.5,bbox=dict(facecolor='#eeeaf4',edgecolor='none',pad=1.5))
    fig.text(.012,.357,'C',fontsize=11,fontweight='bold')
    fig.text(.06,.357,'Complete-pipeline truth recovery',fontsize=9.2,fontweight='bold')
    fig.text(.06,.333,'81 pipelines; 2,550 cohorts; factor-balanced summaries',fontsize=7.2)
    handles=[Line2D([],[],color=STYLES[s][0],marker=STYLES[s][1],linestyle=STYLES[s][2],linewidth=1,markersize=3,label=s.capitalize() if s!='ALL' else 'ALL') for s in SEPS]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.078,.955),ncol=4,columnspacing=1.3,handlelength=2,borderaxespad=0)
    for i,field in enumerate(FIELDS):
        ax=fig.add_subplot(gsA[0,i]);style_axes(ax)
        for s in SEPS:
            z=a[a.separation==s].sort_values('D');color,marker,line=STYLES[s]
            y=z[field]*(100 if field=='extreme_error' else 1)
            ax.plot(z.D,y,color=color,marker=marker,linestyle=line,linewidth=1.25 if s=='ALL' else .9,markersize=3,zorder=5 if s=='ALL' else 3)
        ax.set_title(TITLES[i],pad=5);ax.set_xticks([50,100,150,200,250]);ax.tick_params(axis='x',labelsize=7,rotation=0)
        ax.margins(x=.045,y=.13)
    fig.text(.54,.679,'Retained domain D (μM)',ha='center',fontsize=8)
    for i,(field,title,threshold) in enumerate([('score_spearman','Spearman score fidelity',.98),('class_balanced_label_fidelity','Class-balanced label fidelity',.95)]):
        ax=fig.add_subplot(gsB[0,i]);style_axes(ax)
        ax.axvspan(4.85,5.15,color='#eeeaf4',zorder=0)
        for s in SEPS:
            z=b[b.separation==s].sort_values('K');color,marker,line=STYLES[s]
            ax.plot(z.K,z[field],color=color,marker=marker,linestyle=line,linewidth=1.25 if s=='ALL' else .9,markersize=3,zorder=5 if s=='ALL' else 3)
        lo=min(threshold,b[field].min());hi=max(1,b[field].max());span=hi-lo
        ax.set_ylim(lo-span*.13,hi+span*.16)
        ax.axhline(threshold,color='black',linewidth=.75,linestyle=(0,(4,3)),zorder=2)
        ax.text(6.92,threshold+span*.06,f'{threshold:.2f} reference',ha='right',va='bottom',fontsize=7)
        ax.set_title(title,pad=5);ax.set_xticks([3,4,5,6,7]);ax.margins(x=.045)
    fig.text(.54,.395,'Response-evaluation nodes K',ha='center',fontsize=8)
    # Caption explains K5 highlighting without making a new HB selection claim.
    ordering=[]
    for i,(field,title) in enumerate(zip(['ARI','BA','MacroF1','Extreme_LH'],TITLES)):
        ax=fig.add_subplot(gsC[0,i]);style_axes(ax);ax.grid(False)
        z=c.sort_values([field,'method_id'],ascending=[field=='Extreme_LH',True],na_position='last').copy()
        z['ordinal_position']=np.arange(1,82)
        scale=100 if field=='Extreme_LH' else 1
        z['display_value']=z[field]*scale
        ordering.extend(z[['method_id','ordinal_position','display_value']].assign(outcome=field).to_dict('records'))
        finite=np.isfinite(z.display_value)
        neutral=finite & ~z.method_id.isin([TRIOS,COMPARATOR])
        ax.scatter(z.loc[neutral,'display_value'],z.loc[neutral,'ordinal_position'],s=2.2,color='#454545',zorder=2)
        for mid,color,marker,size,label,ylabel in [('TRIOS','#6A2383','*',33,'TRIOS',-24),(COMPARATOR,'#0072B2','s',13,'Emax–$\\mathrm{EC}_{50}$–\ntertiles',-11)]:
            row=z[z.method_id==mid].iloc[0]
            if np.isfinite(row.display_value):
                ax.scatter([row.display_value],[row.ordinal_position],color=color,marker=marker,s=size,zorder=5)
                ax.annotate(label,xy=(row.display_value,row.ordinal_position),xytext=(.48,ylabel),textcoords=('axes fraction','data'),ha='center',va='center',fontsize=7,color='black',arrowprops=dict(arrowstyle='-',color=color,lw=.6,shrinkA=1,shrinkB=3))
        ax.set_ylim(83,-29);ax.set_title(title,pad=5)
        ax.set_yticks([1,40,81]);ax.margins(x=.12);ax.xaxis.set_major_locator(MaxNLocator(3))
        if i!=0:ax.set_yticklabels([])
    fig.text(.012,.165,'Pipelines ordered by outcome',rotation=90,va='center',fontsize=8)
    fig.text(.54,.027,'Outcome value (extreme error in %; lower is better)',ha='center',fontsize=8)
    pd.DataFrame(ordering).to_csv(HERE/'panel_C_ordered_points.csv',index=False,float_format='%.17g')
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'Figure4_Simulation.{ext}',dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Replay sealed plotting tables only.')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    DATA=Path(__file__).resolve().parent/'data'
    HERE=Path(args.out).resolve();HERE.mkdir(parents=True,exist_ok=True)
    AUDIT=HERE/'replay_qa';AUDIT.mkdir(exist_ok=True)
    read=lambda name:pd.read_csv(DATA/name,float_precision='round_trip')
    draw(read('panel_A.csv'),read('panel_B.csv'),read('panel_C.csv'))
