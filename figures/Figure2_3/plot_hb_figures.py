"""Publication figures from immutable display extracts; no scientific re-estimation.

Contract: Figure 2 is a schematic-led composite linking conditional domain adequacy
to regimen-specific resolution. Figure 3 is a quantitative grid connecting the
113 frozen decisions to their actual fitted functions and ORIGINAL5 benchmark.
Exports: 136 mm full-text width, vector PDF/SVG, 300 dpi PNG, 7 pt minimum text.
Points are profiles, not independent patients. Curve medians are descriptive
pointwise summaries; no confidence intervals or new statistical tests are drawn.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

P=Path(os.environ['TRIOS_FIGURE_OUTPUT']).resolve()
assert json.loads((P/'PARITY_SUMMARY.json').read_text())['status']=='PASS'
def read(n): return pd.read_csv(P/n,float_precision='round_trip')
a=read('figure2_domains.csv');b=read('figure2_fidelity.csv')
s=read('figure3_scores_labels.csv');curves=read('figure3_sampled_frozen_curves.csv');c=read('figure3_benchmark.csv')
REG=['5FU','CARBO','CARBO_A','CDDP','CDDP_A','ETO']
LAB=['5-FU','Carbo','Carbo+A','CDDP','CDDP+A','ETO']
DS=[50,64,64,40,25,50]; KM=[3,4,4,3,6,5]
INK='#17222B';TEAL='#007D80';PURPLE='#69448B';GREY='#788590'
GROUP={'L':'#416EA6','I':'#B77916','H':'#0B806C'}
MARK={'L':'o','I':'s','H':'^'};STYLE={'L':(0,(4,1.5)),'I':(0,(1,1)),'H':'-'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
 'axes.titlesize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'text.color':INK,
 'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,
 'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})
def title(fig,y,letter,text):
    fig.text(.012,y,letter,fontsize=11,fontweight='bold',va='top')
    fig.text(.06,y-.001,text,fontsize=9,fontweight='bold',va='top')
def export(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(P/f'{name}.{ext}',dpi=300)
    plt.close(fig)

fig=plt.figure(figsize=(136/25.4,6.6))
title(fig,.985,'A','HB-PDO study design')
arch=fig.add_axes([.018,.672,.964,.274]);arch.axis('off')
arch.text(.5,.985,'113 eligible PDO–regimen profiles · frozen response fits',ha='center',va='top',fontsize=8,fontweight='bold')
def box(x,y,w,h,head,body):
    arch.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',fc='#F2F6F8',ec='#657A89',lw=.8))
    arch.text(x+.025,y+h-.05,head,fontsize=7.8,fontweight='bold',va='top')
    arch.text(x+.025,y+h-.18,body,fontsize=7,va='top',linespacing=1.32)
box(.015,.525,.425,.34,'Domain selection','Stage attainment + terminal gain\nCandidate endpoints for each regimen')
box(.57,.525,.415,.34,'Complete-pipeline evaluation','CRS + Admissible Breaks L/I/H\n81 pipelines; original five components')
box(.015,.025,.425,.34,'Finite-node resolution','Within selected '+r'$D_r$'+': score / stratum fidelity\n'+r'$K=3$'+'–7; production '+r'$K_{\mathrm{ref}}=7$')
box(.57,.025,.415,.34,'Stage-rule adequacy',r'$R_{1/2}$'+' vs '+r'$R_{0.75}$'+'; operational criterion\n'+r'$\pi_{\mathrm{MS}}=1/2$'+' conditional on adequacy')
def arrow(xy1,xy2,style='-|>'):
    arch.add_patch(FancyArrowPatch(xy1,xy2,arrowstyle=style,mutation_scale=9,lw=.9,color=TEAL))
arrow((.445,.70),(.565,.70));arch.text(.501,.75,'domains',ha='center',fontsize=7)
arrow((.78,.515),(.78,.375),'<->');arch.text(.765,.444,'evaluation',ha='right',fontsize=7)
arrow((.56,.19),(.445,.19));arch.text(.50,.26,'selected '+r'$D_r$',ha='center',fontsize=7)
arch.text(.5,-.03,'Scoring-node resolution is distinct from laboratory dose measurement.',ha='center',fontsize=7)

title(fig,.654,'B','How far? Selected dose domains')
ax=fig.add_axes([.14,.449,.60,.153]);ax.set_ylim(5.65,-.65);ax.set_xlim(0,222)
for i,r in enumerate(REG):
    x=float(a[(a.regimen==r)&(a.rule=='R1/2')].D.iloc[0]);y=float(a[(a.regimen==r)&(a.rule=='R0.75')].D.iloc[0])
    ax.plot([x,y],[i,i],color='#BCC4CB',lw=1.8)
    ax.plot(y,i,'D',ms=4,mfc='white',mec=PURPLE,mew=1)
    ax.plot(x,i,'o',ms=3.8,color=TEAL)
    if x==y: ax.text(x+7,i,str(int(x)),va='center',fontsize=7)
    else:
        ax.text(x-6,i,str(int(x)),ha='right',va='center',color=TEAL,fontsize=7)
        ax.text(y+6,i,str(int(y))+('*' if r=='CDDP_A' else ''),va='center',color=PURPLE,fontsize=7)
ax.set_yticks(range(6),LAB);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel(r'Selected endpoint $D_r$ ($\mu$M)',labelpad=2)
ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False);ax.grid(axis='x',lw=.35,color='#DBE2E7');ax.set_axisbelow(True)
leg=fig.add_axes([.77,.442,.225,.165]);leg.axis('off')
leg.plot([.03],[.94],'o',color=TEAL,ms=4,transform=leg.transAxes);leg.text(.14,.94,r'$R_{1/2}$ (selected)',transform=leg.transAxes,va='center',fontsize=7)
leg.plot([.03],[.73],'D',mfc='white',mec=PURPLE,ms=4,transform=leg.transAxes);leg.text(.14,.73,r'$R_{0.75}$',transform=leg.transAxes,va='center',fontsize=7)
leg.text(0,.50,'Beyond observed\nsupport (profiles):\n'+r'$R_{1/2}$: 0/113'+'\n'+r'$R_{0.75}$: 1/113*',transform=leg.transAxes,fontsize=7,va='top',linespacing=1.4)

title(fig,.385,'C','How much resolution? Fidelity on each selected domain')
fig.text(.205,.340,r'Score fidelity ($\rho_S$)',fontsize=7.8,fontweight='bold')
fig.text(.605,.340,'Class-balanced stratum fidelity',fontsize=7.6,fontweight='bold')
fig.text(.205,.315,r'$\eta_S=0.98$',fontsize=7);fig.text(.605,.315,r'$\eta_G=0.95$',fontsize=7)
cmap=LinearSegmentedColormap.from_list('fidelity',['#FCE5CA','#DBEAF0','#86BBC5']);norm=Normalize(.90,1.)
for col,field,thresh in [(0,'F_score',.98),(1,'F_strata',.95)]:
    xx=.205+col*.40
    aa=fig.add_axes([xx,.112,.34,.186])
    mat=np.array([[float(b[(b.regimen==r)&(b.K==k)][field].iloc[0]) for k in range(3,8)] for r in REG])
    aa.pcolormesh(np.arange(6)-.5,np.arange(7)-.5,mat,cmap=cmap,norm=norm,rasterized=False)
    aa.set_xlim(-.5,4.5);aa.set_ylim(5.5,-.5)
    for i in range(6):
        for j in range(5):
            aa.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=7)
            if mat[i,j]<thresh: aa.plot([j-.31,j+.31],[i+.29,i+.29],color='#8F411E',lw=1.15)
        aa.add_patch(Rectangle((KM[i]-3-.46,i-.46),.92,.92,fill=False,ec=TEAL,lw=1.3))
    aa.set_xticks(range(5),['3','4','5','6','7']);aa.set_xlabel(r'Scoring nodes $K$',labelpad=2)
    aa.set_yticks(range(6),[f'{lab} ({d})' for lab,d in zip(LAB,DS)] if col==0 else ['']*6)
    aa.tick_params(length=0,pad=3);aa.set_xticks(np.arange(-.5,5,1),minor=True);aa.set_yticks(np.arange(-.5,6,1),minor=True)
    aa.grid(which='minor',color='white',lw=.8);aa.tick_params(which='minor',length=0)
    for sp in aa.spines.values():sp.set_visible(False)
fig.text(.015,.308,r'Regimen ($D_r$, $\mu$M)',fontsize=7)
fig.text(.975,.314,r'$K_{\mathrm{MS},r}$',fontsize=7,ha='center')
for i,k in enumerate(KM):fig.text(.975,.112+.186*(1-(i+.5)/6),str(k),fontsize=7.5,fontweight='bold',ha='center',va='center',color=TEAL)
fig.text(.205,.048,r'Reference $K_{\mathrm{ref}}=7$; common design $K_{\mathrm{MS}}^{\mathrm{HB}}=6$.',fontsize=7.5)
fig.text(.205,.020,'Outlines: minimum adequate tail; underlining: below threshold.',fontsize=7)
export(fig,'Figure2_HB')

fig=plt.figure(figsize=(136/25.4,7.30))
title(fig,.99,'A','Frozen seven-node CRS and ordered strata')
handles=[Line2D([0],[0],color=GROUP[g],marker=MARK[g],linestyle=STYLE[g],lw=1.3,ms=3,label=l) for g,l in zip('LIH',['Low','Intermediate','High'])]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.58,.970),ncol=3,frameon=False,columnspacing=1.3,handlelength=1.5)
ymin=min(0,float(s.score.min())-.035);ymax=max(1,float(s.score.max())+.035)
for i,r in enumerate(REG):
    ax=fig.add_axes([.105+(i%3)*.315,.803-(i//3)*.151,.248,.102])
    ss=s[s.regimen==r].sort_values(['score','profile_id']);x=np.arange(1,len(ss)+1)
    for g in 'LIH':
        sel=ss.label.eq(g).to_numpy();ax.scatter(x[sel],ss.score.to_numpy()[sel],marker=MARK[g],s=10,color=GROUP[g],linewidths=.2)
    for t in ['tau1','tau2']:ax.axhline(ss[t].iloc[0],lw=.65,ls='--',color='#3F4850',zorder=0)
    cnt=tuple(int(ss.label.eq(g).sum()) for g in 'LIH')
    ax.set_title(f'{LAB[i]}  '+r'$n$'+f'={len(ss)}; L/I/H={cnt[0]}/{cnt[1]}/{cnt[2]}',fontsize=7.1,pad=3,loc='left')
    ax.set_xlim(.3,len(ss)+.7);ax.set_ylim(ymin,ymax);ax.set_xticks([1,len(ss)]);ax.set_yticks([0,.5,1]);ax.tick_params(length=2,pad=2)
    if i%3==0:ax.set_ylabel('CRS',labelpad=2)
    if i//3==1:ax.set_xlabel('Profiles (ascending CRS)',labelpad=2)
title(fig,.590,'B','Frozen response curves within selected domains')
fig.text(.105,.561,'Thin: individual profiles; thick: pointwise stratum medians.',fontsize=7)
y0=float(curves.y_fitted.min());y1=float(curves.y_fitted.max());pad=(y1-y0)*.04
for i,r in enumerate(REG):
    ax=fig.add_axes([.105+(i%3)*.315,.433-(i//3)*.155,.248,.102])
    cc=curves[curves.regimen==r]
    for (pid,g),cv in cc.groupby(['profile_id','label'],sort=False):
        ax.plot(cv.dose_uM,cv.y_fitted,color=GROUP[g],alpha=.35,lw=.48,ls=STYLE[g])
    for g in 'LIH':
        med=cc[cc.label==g].groupby('dose_uM',sort=True).y_fitted.median()
        ax.plot(med.index,med.values,color=GROUP[g],lw=1.45,ls=STYLE[g])
    ax.set_xscale('function',functions=(np.log1p,np.expm1));ax.set_xlim(0,DS[i]);ax.set_ylim(y0-pad,y1+pad)
    ax.set_xticks([0,5,DS[i]],[str(z) for z in [0,5,DS[i]]]);ax.set_yticks([0,.5,1]);ax.tick_params(length=2,pad=2)
    ax.set_title(f'{LAB[i]}  '+r'$D_r$'+f'={DS[i]} '+r'$\mu$M',fontsize=7.4,pad=3,loc='left')
    if i%3==0:ax.set_ylabel('Fitted inhibition',labelpad=2)
    if i//3==1:ax.set_xlabel(r'Dose ($\mu$M; log1p spacing)',labelpad=2)
title(fig,.214,'C','Operational comparison of 81 complete pipelines')
ax=fig.add_axes([.105,.074,.85,.106])
for row in c.sort_values(['rank','method_id']).itertuples():
    ax.scatter(row.rank,row.C_operational,s=10,facecolors=GREY if row.A_valid==1 else 'white',edgecolors=GREY,lw=.65,zorder=2)
tr=c[c.method_id=='TRIOS'].iloc[0];co=c[c.method_id=='EMAX|EC50|TERTILES'].iloc[0]
for row,color in [(tr,TEAL),(co,PURPLE)]:ax.scatter(row['rank'],row.C_operational,s=26,color=color,zorder=4)
ax.annotate('TRIOS: 1/81; 0.8943',xy=(tr['rank'],tr.C_operational),xytext=(17,.91),fontsize=7,color=TEAL,va='center',arrowprops=dict(arrowstyle='-',color=TEAL,lw=.7))
ax.annotate(r'Emax–$\mathrm{EC}_{50}$–tertiles: 0.7493',xy=(co['rank'],co.C_operational),xytext=(19,.35),fontsize=7,color=PURPLE,va='center',arrowprops=dict(arrowstyle='-',color=PURPLE,lw=.7))
ax.set_xlim(-1,83);ax.set_ylim(-.04,1.08);ax.set_xticks([1,20,40,60,81]);ax.set_yticks([0,.5,1]);ax.set_xlabel('Operational rank among 81 pipelines',labelpad=2);ax.set_ylabel(r'$C_{\mathrm{operational}}$',labelpad=2)
ax.grid(axis='y',color='#DFE5EA',lw=.4);ax.set_axisbelow(True);ax.tick_params(pad=2,length=2)
fig.text(.105,.018,r'Filled: $A_{\mathrm{valid}}=1$; open: $A_{\mathrm{valid}}<1$. Rank ties retained.',fontsize=7)
export(fig,'Figure3_HB')
print('Exported Figures 2 and 3 as vector PDF/SVG and 300 dpi PNG.')
