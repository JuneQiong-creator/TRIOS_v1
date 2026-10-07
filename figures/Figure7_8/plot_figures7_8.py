from pathlib import Path

import hashlib, json, platform, re, io, base64

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from matplotlib import font_manager

from fontTools.ttLib import TTFont

from fontTools import subset

ORDER = ['CCLE-34-Paclitaxel-SKIN', 'CCLE-11-AZD6244-LUNG', 'CCLE-38-PD-0325901-HEM',
         'CCLE-10-AZD6244-HEM', 'CCLE-65-Topotecan-LUNG', 'CCLE-51-PLX4720-SKIN', 'CCLE-58-TAE684-HEM']

LABELS = ['Paclitaxel–SKIN', 'AZD6244–LUNG', 'PD-0325901–HEM', 'AZD6244–HEM',
          'Topotecan–LUNG', 'PLX4720–SKIN', 'TAE684–HEM']

FEATURES = ['FBXW7-related\nmitotic-apoptosis\nmodule', 'Canonical MAPK\nactivation',
            'Canonical MAPK\nactivation', 'Canonical MAPK\nactivation', 'SLFN11 expression',
            'BRAF V600', 'ALK rearrangement']

WIDTH_MM = 136

F7_HEIGHT_MM, F8_HEIGHT_MM = 156, 148

BLUE, TEAL, PURPLE, ORANGE = '#0072B2', '#009E73', '#6A2383', '#D55E00'

STRATA = {'L': '#416EA6', 'I': '#B77916', 'H': '#0B806C'}

DOSES = [.0025,.008,.025,.08,.25,.8,2.53,8.]

INPUTS, SOURCE_MAP, QA = {}, [], []

def style():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
                        'axes.linewidth':.65,'text.color':'black','axes.labelcolor':'black',
                        'xtick.color':'black','ytick.color':'black','legend.fontsize':7,'legend.frameon':False,
                        'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7,length=2.5,width=.65,pad=2)


def embed_svg_fonts(path):
    import xml.etree.ElementTree as ET
    svg = path.read_text(encoding='utf-8')
    chars = ''.join(ET.fromstring(svg).itertext())
    rules = []
    for family,weight,style_ in [('Arial','normal','normal'),('Arial','bold','normal'),('DejaVu Sans','normal','normal'),('DejaVu Sans','normal','oblique')]:
        fp = Path(font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style_)))
        font=TTFont(fp); options=subset.Options(); options.drop_tables += ['FFTM','meta']
        sub=subset.Subsetter(options=options); sub.populate(text=chars); sub.subset(font)
        font.flavor='woff'; stream=io.BytesIO(); font.save(stream); font.close()
        rules.append(f"@font-face {{font-family:'{family}';font-style:{style_};font-weight:{'700' if weight=='bold' else '400'};src:url(data:font/woff;base64,{base64.b64encode(stream.getvalue()).decode('ascii')}) format('woff');}}")
    css='<style type="text/css"><![CDATA[\n'+'\n'.join(rules)+'\n]]></style>\n'
    path.write_text(re.sub(r'(<defs>)',lambda m:m.group(1)+'\n'+css,svg,count=1),encoding='utf-8')


def save(fig,name):
    # Exact physical canvas size, not bbox_inches='tight' rescaling.
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white')
    embed_svg_fonts(HERE/f'{name}.svg')
    plt.close(fig)


def plot7(d):
    fig = plt.figure(figsize=(WIDTH_MM/25.4,F7_HEIGHT_MM/25.4),facecolor='white')
    y = np.arange(7)
    axa = fig.add_axes([.255,.636,.363,.274])
    axb = fig.add_axes([.699,.636,.246,.274])
    for ax in [axa,axb]:
        clean(ax); ax.set_ylim(6.5,-.5); ax.set_yticks(y)
    fig.text(.015,.981,'A',weight='bold',fontsize=11,va='top')
    fig.text(.255,.981,'Pharmacodynamic\ndomain',weight='bold',fontsize=8.5,va='top')
    fig.text(.660,.981,'B',weight='bold',fontsize=11,va='top')
    fig.text(.705,.981,'Ordered stratum\ncomposition',weight='bold',fontsize=8.5,va='top')
    axa.set_yticklabels(LABELS,fontsize=7)
    axa.set_xscale('log'); axa.set_xlim(.0018,28)
    doses=DOSES
    axa.set_xticks(doses,[f'{x:g}' for x in doses],rotation=55,ha='right')
    axa.minorticks_off(); axa.grid(axis='x',color='#dddddd',linewidth=.4)
    axa.axvline(8,color='black',lw=.8,ls=(0,(3,2)))
    axa.scatter(d.D_transport,y,s=20,color=TEAL,zorder=3)
    for yy,val in zip(y,d.D_transport):
        axa.annotate(f'{val:g}',(val,yy),xytext=(-5 if val==2.53 else 5,0),textcoords='offset points',
                     ha='right' if val==2.53 else 'left',va='center',fontsize=7,
                     bbox={'facecolor':'white','edgecolor':'none','pad':.3})
    axa.set_xlabel('Selected domain (µM)',labelpad=3)
    axb.set_yticklabels([]); axb.tick_params(axis='y',length=0)
    left=np.zeros(7)
    for s,color in STRATA.items():
        vals=d['proportion_'+s].to_numpy()*100
        axb.barh(y,vals,left=left,height=.52,color=color)
        for yy,x,n in zip(y,left+vals/2,d['n_'+s]):
            axb.text(x,yy,str(n),fontsize=7,color='white',ha='center',va='center')
        left += vals
    axb.set_xlim(0,100); axb.set_xticks([0,50,100],['0','50','100'])
    axb.set_xlabel('Profiles (%)',labelpad=3)
    for yy,n in zip(y,d.N):
        axb.text(104,yy,str(n),fontsize=7,ha='left',va='center',clip_on=False)
    axb.text(104,-.66,'N',fontsize=7,ha='left',va='bottom',clip_on=False)
    axb.legend(handles=[Patch(facecolor=STRATA[s],label=s) for s in STRATA],loc='lower left',bbox_to_anchor=(-.10,1.015),
               ncol=3,handlelength=.75,columnspacing=.85,handletextpad=.35,borderaxespad=0)
    # Molecular row: true quantitative axis plus separate aligned annotation columns.
    fig.text(.015,.538,'C',weight='bold',fontsize=11,va='top')
    fig.text(.060,.538,'Primary molecular associations',weight='bold',fontsize=8.5,va='top')
    axc=fig.add_axes([.478,.090,.176,.365]); clean(axc)
    axc.set_xlim(-1,1);axc.set_ylim(6.5,-.5);axc.set_yticks([])
    axc.set_xticks([-1,0,1]);axc.axvline(0,color='#666666',lw=.75,ls=(0,(3,2)))
    axc.set_xlabel("Cliff’s δ",labelpad=4)
    cols=[(.015,'Drug–lineage'),(.248,'Primary feature'),(.70,'Positive /\ncall-negative'),(.803,'Expected\nCRS'),(.928,'q / status')]
    for xx,label in cols:fig.text(xx,.488,label,ha='left' if xx<.5 else 'center',va='center',fontsize=7,fontweight='bold')
    for i,row in enumerate(d.itertuples()):
        yy=.090+.365*(1-(i+.5)/7)
        fig.text(.015,yy,LABELS[i],fontsize=7,va='center')
        fig.text(.248,yy,FEATURES[i],fontsize=7,va='center',linespacing=1.15)
        is_formal=row.evaluability=='INFERENTIALLY_EVALUABLE'
        if np.isfinite(row.delta):
            axc.plot(row.delta,i,marker='o',ls='None',ms=4.4,mew=1,mec=PURPLE,mfc=PURPLE if is_formal else 'white')
        counts='—' if pd.isna(row.n_positive) else f'{int(row.n_positive)}/{int(row.n_call_negative)}'
        fig.text(.70,yy,counts,ha='center',va='center',fontsize=7)
        fig.text(.803,yy,'Lower' if row.expected_direction=='LOWER_CRS' else 'Higher',ha='center',va='center',fontsize=7)
        status=f'{row.q:.5f}' if is_formal else ('Descriptive;\n2 positive' if row.evaluability=='SPARSE_DESCRIPTIVE_ONLY' else 'Not evaluable')
        fig.text(.928,yy,status,ha='center',va='center',fontsize=7,linespacing=1.2)
    fig.text(.566,.022,'Feature-positive minus call-negative',ha='center',fontsize=7)
    save(fig,'Figure7_CCLE_Transport_Molecular')


def plot8(d,g):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,F8_HEIGHT_MM/25.4),facecolor='white')
    axs=[fig.add_axes([.255,.554,.326,.338]),fig.add_axes([.641,.554,.326,.338])]
    for panel,scope,ax in zip(['A','B'],['FULL','B100'],axs):
        clean(ax);ax.set_ylim(6.5,-.5);ax.set_xlim(-.02,1.0);ax.set_xticks([0,.25,.5,.75,1.])
        ax.set_yticks(np.arange(7));ax.set_yticklabels(LABELS if panel=='A' else [],fontsize=7)
        if panel=='B':ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#dddddd',linewidth=.4)
        ax.set_xlabel(r'$C_{\mathrm{operational}}$',labelpad=4)
        for i,cohort in enumerate(ORDER):
            z=d[d.dataset_id.eq(cohort)&d.scope.eq(scope)]
            conv=z[~z.method_id.eq('TRIOS')];tr=z[z.method_id.eq('TRIOS')].iloc[0]
            ax.scatter(conv.C_operational,i+conv.display_y_offset,s=6,color='#555555',alpha=.80,linewidths=0,zorder=2)
            best=conv[conv.best_conventional]
            ax.scatter(best.C_operational,i+best.display_y_offset,s=21,marker='s',facecolors='white',edgecolors=ORANGE,linewidths=1,zorder=3)
            ax.scatter(tr.C_operational,i,s=58,marker='*',color=TEAL,edgecolors='black',linewidths=.3,zorder=4)
    fig.text(.015,.981,'A',weight='bold',fontsize=11,va='top')
    fig.text(.255,.981,'Full-source cohorts',weight='bold',fontsize=8.5,va='top')
    fig.text(.607,.981,'B',weight='bold',fontsize=11,va='top')
    fig.text(.648,.981,'Size-18 subsets:\naggregate quality',weight='bold',fontsize=8.5,va='top')
    legend=[Line2D([],[],marker='o',color='#555555',ms=3,ls='None',label='Conventional'),
            Line2D([],[],marker='*',color=TEAL,mec='black',mew=.3,ms=7,ls='None',label='TRIOS'),
            Line2D([],[],marker='s',color=ORANGE,mfc='white',ms=4,ls='None',label='Best conventional (ties)')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.576,.940),ncol=3,handletextpad=.45,columnspacing=1,borderaxespad=0)
    fig.text(.015,.444,'C',weight='bold',fontsize=11,va='top')
    fig.text(.255,.444,'Best-conventional gaps',weight='bold',fontsize=8.5,va='top')
    ax=fig.add_axes([.255,.073,.710,.314]);clean(ax)
    ax.set_ylim(6.5,-.5);ax.set_xlim(-.004,.13);ax.set_xticks([0,.025,.05,.075,.10,.125])
    ax.set_yticks(np.arange(7),LABELS,fontsize=7);ax.axvline(0,color='black',lw=.75,ls=(0,(3,2)))
    ax.grid(axis='x',color='#dddddd',linewidth=.4)
    for i,cohort in enumerate(ORDER):
        z=g[g.dataset_id.eq(cohort)].set_index('scope')
        vals=[z.loc[s,'gap'] for s in ['FULL','B100']]
        ax.plot(vals,[i-.085,i+.085],color='#888888',lw=.6,zorder=1)
        ax.plot(vals[0],i-.085,'o',color=BLUE,ms=4.2,ls='None',zorder=3)
        ax.plot(vals[1],i+.085,'s',color=PURPLE,ms=4.2,ls='None',zorder=3)
    ax.set_xlabel(r'$\Delta C_{\mathrm{operational}}$ (TRIOS − best conventional)',labelpad=4)
    ax.legend(handles=[Line2D([],[],marker='o',color=BLUE,ms=4,ls='None',label='Full-source'),
                       Line2D([],[],marker='s',color=PURPLE,ms=4,ls='None',label='Size-18 aggregate')],
              loc='lower right',bbox_to_anchor=(1,1.035),ncol=2,handletextpad=.4,columnspacing=1,borderaxespad=0)
    save(fig,'Figure8_CCLE_Comparative')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Replay sealed plotting tables only.')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    DATA=Path(__file__).resolve().parent/'data'
    HERE=Path(args.out).resolve();HERE.mkdir(parents=True,exist_ok=True)
    AUDIT=HERE/'replay_qa';AUDIT.mkdir(exist_ok=True)
    read=lambda name:pd.read_csv(DATA/name,float_precision='round_trip')
    style();plot7(read('Figure7_source_data.csv'));plot8(read('Figure8_source_data.csv'),read('Figure8_gap_data.csv'))
