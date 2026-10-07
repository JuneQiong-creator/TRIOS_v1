from pathlib import Path

import hashlib,json,zipfile,platform,re,base64,io

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import font_manager

from fontTools.ttLib import TTFont

from fontTools import subset

WIDTH_MM,HEIGHT_MM=136,175

GROUP_NAMES={'A':'Local non-monotonicity','B':'Alternative noise distributions','C':'Potency–efficacy conflict','D':'Delayed pharmacodynamic stages'}

COLORS={'A':'#0072B2','B':'#D55E00','C':'#6A2383','D':'#009E73'}

MARKERS={'A':'o','B':'s','C':'D','D':'^'}

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

INPUTS={}

CHECKS=[]

def embed_svg_fonts(path):
    import xml.etree.ElementTree as ET
    svg=path.read_text(encoding='utf-8'); chars=''.join(ET.fromstring(svg).itertext())
    styles=[('Arial','normal','normal'),('Arial','bold','normal'),('DejaVu Sans','normal','normal'),('DejaVu Sans','normal','oblique')]
    rules=[];fonts=[]
    for family,weight,style in styles:
        fp=Path(font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style)))
        font=TTFont(fp); options=subset.Options();options.drop_tables+=['FFTM','meta']
        sub=subset.Subsetter(options=options);sub.populate(text=chars);sub.subset(font)
        font.flavor='woff'; stream=io.BytesIO();font.save(stream);font.close()
        encoded=base64.b64encode(stream.getvalue()).decode('ascii')
        rules.append(f"@font-face {{font-family:'{family}';font-style:{style};font-weight:{'700' if weight=='bold' else '400'};src:url(data:font/woff;base64,{encoded}) format('woff');}}")
        fonts.append(dict(family=family,weight=weight,style=style,path=str(fp),sha256=sha(fp),embedded_subset_bytes=len(stream.getvalue())))
    css='<style type="text/css"><![CDATA[\n'+'\n'.join(rules)+'\n]]></style>\n'
    svg=re.sub(r'(<defs>)',lambda m:m.group(1)+'\n'+css,svg,count=1)
    path.write_text(svg,encoding='utf-8')
    (AUDIT/'SVG_FONT_MANIFEST.json').write_text(json.dumps(fonts,indent=2),encoding='utf-8')


def plot(data):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'axes.linewidth':.65,
                        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
                        'legend.fontsize':7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),facecolor='white')
    row_headers=[.982,.694,.450,.219]
    row_tops=[.892,.662,.418,.187]
    row_bottoms=[.775,.545,.301,.070]
    row_xlabels=[.728,.487,.255,.025]
    fields=['A_valid','ARI','Extreme_percent']
    titles=[r'Applicability $A_{\mathrm{valid}}$','Adjusted Rand\nindex','Extreme L↔H\nerror (%) ↓']
    limits=[(.964,1.003),(.405,.545),(.20,.95)]
    ticks=[[.97,.98,.99,1.],[.42,.46,.50,.54],[.2,.4,.6,.8]]
    xlabels=['Local-dip severity','Noise distribution','Conflict severity','Delay multiplier']
    axis_qa=[]
    for row,panel in enumerate(['A','B','C','D']):
        z=data[data.panel.eq(panel)].sort_values('scenario_order')
        assert len(z)==3
        fig.text(.012,row_headers[row],panel,fontsize=11,fontweight='bold',va='top')
        fig.text(.060,row_headers[row],GROUP_NAMES[panel],fontsize=9.2,fontweight='bold',va='top')
        gs=fig.add_gridspec(1,3,left=.108,right=.985,bottom=row_bottoms[row],top=row_tops[row],wspace=.43)
        for col,field in enumerate(fields):
            ax=fig.add_subplot(gs[0,col]);ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=7,length=2.5,width=.65,pad=2,colors='black')
            ax.set_ylim(*limits[col]);ax.set_yticks(ticks[col]);ax.grid(axis='y',linewidth=.4,color='#dddddd',zorder=0)
            y=z[field].to_numpy();assert np.all((y>=limits[col][0])&(y<=limits[col][1]))
            x=np.arange(3) if panel!='D' else np.array([float(v) for v in z.frozen_severity_or_parameter])
            if panel=='B':
                ax.plot(x,y,color=COLORS[panel],marker=MARKERS[panel],markersize=3.5,ls='None')
                labels=['Hetero-\nscedastic','Heavy-\ntailed $t_3$','Contam-\ninated']
            else:
                ax.plot(x,y,color=COLORS[panel],marker=MARKERS[panel],markersize=3.5,lw=1)
                labels=list(z.display_scenario)
            ax.set_xticks(x,labels);ax.margins(x=.18)
            if row==0:ax.set_title(titles[col],pad=7)
            axis_qa.append(dict(panel=panel,field=field,ymin=limits[col][0],ymax=limits[col][1],ticks=';'.join(map(str,ticks[col])),point_count=3,line_connected=panel!='B'))
        fig.text(.545,row_xlabels[row],xlabels[row],ha='center',fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'Figure6_Structural_Stress.{ext}',dpi=300,facecolor='white')
    embed_svg_fonts(HERE/'Figure6_Structural_Stress.svg')
    pd.DataFrame(axis_qa).to_csv(AUDIT/'AXIS_QA.csv',index=False)
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
    plot(read('Figure6_PLOT_DATA.csv'))
