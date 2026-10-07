from pathlib import Path

import hashlib,json,zipfile,platform,base64,io,re

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator,FormatStrFormatter

from matplotlib import font_manager

from fontTools.ttLib import TTFont

from fontTools import subset

WIDTH_MM,HEIGHT_MM=136,165

COLORS={'weak':'#0072B2','moderate':'#D55E00','strong':'#009E73','ALL':'#000000'}

MARKERS={'weak':'o','moderate':'s','strong':'^','ALL':'D'}

LINES={'weak':'--','moderate':'--','strong':':','ALL':'-'}

hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

INPUTS={}

CHECKS=[]

MAP=[]

def axes_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7,length=2.5,width=.65,pad=2,colors='black')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis='y',linewidth=.4,color='#dddddd',zorder=0)


def embed_svg_fonts(path):
    """Embed the exact font subsets; SVG text remains editable and portable."""
    import xml.etree.ElementTree as ET
    svg=path.read_text(encoding='utf-8')
    chars=''.join(ET.fromstring(svg).itertext())
    styles=[('Arial','normal','normal'),('Arial','bold','normal'),
            ('DejaVu Sans','normal','normal'),('DejaVu Sans','normal','oblique')]
    rules=[]; fonts=[]
    for family,weight,style in styles:
        fontpath=Path(font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style)))
        font=TTFont(fontpath)
        options=subset.Options(); options.drop_tables+=['FFTM']
        sub=subset.Subsetter(options=options); sub.populate(text=chars); sub.subset(font)
        font.flavor='woff'; stream=io.BytesIO(); font.save(stream); font.close()
        encoded=base64.b64encode(stream.getvalue()).decode('ascii')
        rules.append(f"@font-face {{ font-family:'{family}'; font-style:{style}; font-weight:{'700' if weight=='bold' else '400'}; src:url(data:font/woff;base64,{encoded}) format('woff'); }}")
        fonts.append(dict(family=family,weight=weight,style=style,source=str(fontpath),sha256=hashfile(fontpath),embedded_subset_bytes=len(stream.getvalue())))
    css='<style type="text/css"><![CDATA[\n'+'\n'.join(rules)+'\n]]></style>\n'
    svg=re.sub(r'(<defs>)',lambda m:m.group(1)+'\n'+css,svg,count=1)
    path.write_text(svg,encoding='utf-8')
    (AUDIT/'SVG_FONT_MANIFEST.json').write_text(json.dumps(fonts,indent=2),encoding='utf-8')


def plot(a,b,seps,c):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
                        'axes.linewidth':.65,'text.color':'black','axes.labelcolor':'black',
                        'xtick.color':'black','ytick.color':'black','legend.fontsize':7,
                        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),facecolor='white')
    ga=fig.add_gridspec(1,3,left=.10,right=.99,bottom=.687,top=.835,wspace=.40)
    gb=fig.add_gridspec(1,2,left=.10,right=.99,bottom=.364,top=.485,wspace=.26)
    gc=fig.add_gridspec(1,5,left=.105,right=.99,bottom=.067,top=.163,wspace=.52)
    fig.text(.012,.977,'A',fontsize=11,fontweight='bold')
    fig.text(.06,.977,'Upper measured concentration',fontsize=9.2,fontweight='bold')
    fig.text(.06,.954,'Seven positive concentrations; three replicates per condition',fontsize=7.5)
    fig.text(.06,.931,'First operationally eligible support: H=150 μM',fontsize=7.5)
    fig.legend(handles=[Line2D([],[],color='#0072B2',marker='o',lw=1,markersize=3,label='Applicability'),Line2D([],[],color='#6A2383',marker='s',lw=1,markersize=3,label='Operational quality')],bbox_to_anchor=(.06,.913),loc='upper left',ncol=2,handlelength=1.6,columnspacing=1.3,borderaxespad=0)
    fields=[None,'CRS_spearman','balanced_label_fidelity']
    titles=['Applicability and\noperational quality','Spearman score\nfidelity','Class-balanced\nlabel fidelity']
    for i,field in enumerate(fields):
        ax=fig.add_subplot(ga[0,i]);axes_style(ax)
        ax.axvspan(143,157,color='#eeeaf4',zorder=0)
        if i==0:
            ax.plot(a.H,a.A_valid,color='#0072B2',marker='o',markersize=3,lw=1)
            ax.plot(a.H,a.C_operational,color='#6A2383',marker='s',markersize=3,lw=1)
            ax.set_ylim(.835,1.015);threshold=.95
            ax.text(246,.956,'Applicability\nthreshold 0.95',ha='right',va='bottom',fontsize=7)
        else:
            ax.plot(a.H,a[field],color='black',marker='D',markersize=3,lw=1)
            threshold=.98 if i==1 else .95
            lo=min(a[field].min(),threshold);span=1-lo
            ax.set_ylim(lo-span*.13,1+span*.12)
            ax.text(102,threshold+span*.02,f'{threshold:.2f} reference',ha='left',va='bottom',fontsize=7)
        ax.axhline(threshold,color='black',lw=.7,ls=(0,(4,3)),zorder=1)
        ax.set_xticks([100,150,200,250]);ax.margins(x=.06);ax.set_title(titles[i],pad=5)
    fig.text(.54,.649,'Upper measured concentration H (μM)',ha='center',fontsize=8)
    fig.text(.012,.603,'B',fontsize=11,fontweight='bold')
    fig.text(.06,.603,'Measured concentration count',fontsize=9.2,fontweight='bold')
    fig.text(.06,.578,r'H=150 μM; zero-dose control retained; scoring $K_{\mathrm{ref}}=7$',fontsize=7.5)
    fig.text(.06,.555,'First overall eligible count: J=4',fontsize=7.5)
    handles=[Line2D([],[],color=COLORS[s],marker=MARKERS[s],linestyle=LINES[s],lw=1,markersize=3,label='Overall' if s=='ALL' else s.capitalize()) for s in ['weak','moderate','strong','ALL']]
    fig.legend(handles=handles,bbox_to_anchor=(.06,.538),loc='upper left',ncol=4,handlelength=1.6,columnspacing=1.1,borderaxespad=0)
    for i,title in enumerate(['Applicability','Operational quality']):
        ax=fig.add_subplot(gb[0,i]);axes_style(ax);ax.axvspan(3.86,4.14,color='#eeeaf4',zorder=0)
        if i==0:
            for s in ['weak','moderate','strong','ALL']:
                z=b if s=='ALL' else seps[seps.stratum==s].sort_values('J')
                ax.plot(z.J,z.A_valid,color=COLORS[s],marker=MARKERS[s],ls=LINES[s],lw=1.15 if s=='ALL' else .9,markersize=3)
            ax.axhline(.95,color='black',ls=(0,(4,3)),lw=.7)
            ax.text(6.94,.941,'0.95 applicability',fontsize=7,ha='right',va='top')
            lo=min(b.A_valid.min(),seps.A_valid.min());ax.set_ylim(lo-.035,1.03)
        else:
            ax.plot(b.J,b.C_operational,color='black',marker='D',lw=1.15,markersize=3)
            ax.margins(y=.14)
        ax.set_xticks([3,4,5,6,7]);ax.margins(x=.06);ax.set_title(title,pad=4)
        for j,w in zip([3,4,5,6,7],[12,15,18,21,24]):ax.text(j,-.46,str(w),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7)
    fig.text(.54,.326,'Positive measured concentrations J',ha='center',fontsize=8)
    fig.text(.012,.30834,'Wells',fontsize=7,va='top')
    fig.text(.012,.263,'C',fontsize=11,fontweight='bold')
    fig.text(.06,.263,'Paired changes from the reference',fontsize=9.2,fontweight='bold')
    fig.text(.06,.238,'Candidate minus J=7; H=150 μM',fontsize=7.5)
    titles=[r'$\Delta C_{\mathrm{operational}}$','ΔAdjusted\nRand index','ΔBalanced\naccuracy','ΔMacro-F1','ΔExtreme\nL↔H error\n(percentage\npoints)']
    for i,(metric,title) in enumerate(zip(['C_operational_E4','ARI','BA','MacroF1','Extreme'],titles)):
        ax=fig.add_subplot(gc[0,i]);axes_style(ax);ax.grid(False)
        z=c[c.metric==metric].sort_values('J');assert list(z.J)==[4,5,6]
        x=z.display_difference.to_numpy();lo=z.display_CI95_low.to_numpy();hi=z.display_CI95_high.to_numpy()
        assert np.isfinite([x,lo,hi]).all() and np.all(lo<=x) and np.all(x<=hi)
        ax.errorbar(x,[0,1,2],xerr=[x-lo,hi-x],fmt='D',markersize=3,color='#6A2383',ecolor='black',elinewidth=.8,capsize=2,capthick=.7,zorder=3)
        ax.axvline(0,color='black',lw=.7,ls=(0,(4,3)),zorder=1)
        minimum=min(lo.min(),0);maximum=max(hi.max(),0);span=maximum-minimum
        ax.set_xlim(minimum-.13*span,maximum+.13*span);ax.set_ylim(2.5,-.5)
        ax.set_yticks([0,1,2],['J=4','J=5','J=6'] if i==0 else ['', '', ''])
        ax.xaxis.set_major_locator(MaxNLocator(2));ax.tick_params(axis='y',length=0)
        ax.set_title(title,fontsize=8,pad=5)
        fig.text((ax.get_position().x0+ax.get_position().x1)/2,.018,'← Better' if i==4 else 'Better →',ha='center',fontsize=7)
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'Figure5_Prospective_Assay.{ext}',dpi=300,facecolor='white')
    embed_svg_fonts(HERE/'Figure5_Prospective_Assay.svg')
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
    plot(read('panel_A.csv'),read('panel_B_overall.csv'),read('panel_B_separation.csv'),read('panel_C.csv'))
