from d5a import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
def main():
 p=pd.read_csv(OUT/'07_SYNTHESIS/D5_SEVEN_DATASET_PRIMARY_FEATURE_TABLE.csv',float_precision='round_trip');j=pd.read_csv(OUT/'03_CONTINUOUS_CRS/ANALYSIS_JOIN.csv',float_precision='round_trip');cases=pd.read_csv(OUT/'07_SYNTHESIS/D5_DISPLAY_CASES.csv',float_precision='round_trip')
 js('09_FIGURES/FIGURE_CONTRACT.json',{'claim':'Prespecified molecular coherence is reported with coverage limitations and descriptive repeated-subset stability; no overall biology pass/fail claim.','archetype':'quantitative grid','backend':'Python matplotlib','formats':['PNG'],'source':'Frozen current D5 primary tables and B100/OOF summaries','unit':'unique cell line within dataset; overlapping subsets descriptive only','inference':'two-sided permutation; dataset-specific within-family BH; no subset iid inference','display_selection':'two largest absolute primary Cliff delta, ties smaller q; includes sparse primary if selected, explicitly labelled; not a success criterion'})
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.right':False,'axes.spines.top':False})
 labels=[c.replace('CCLE-','').split('-',1)[1] for c in p.cohort_id];yy=np.arange(len(p));png=[]
 def save(fig,name):
  fig.tight_layout(rect=(0,.06,1,.98));path=OUT/'09_FIGURES'/name;fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig);png.append(path)
 fig,ax=plt.subplots(figsize=(12,4.8));ax.axis('off');rows=[]
 for r in p.itertuples():rows.append([r.cohort_id.replace('CCLE-',''),r.feature.replace('_',' '),f'{r.n_positive}/{r.n_call_negative}',r.evaluability.replace('_',' '),f'{r.cliffs_delta:.3f}' if np.isfinite(r.cliffs_delta) else 'NA',f'{r.q:.4g}' if np.isfinite(r.q) else 'Not tested'])
 table=ax.table(cellText=rows,colLabels=['Dataset','Primary feature','+ / call-negative','Evaluability','Cliff delta','q (CRS)'],loc='center',cellLoc='left',colWidths=[.20,.27,.12,.25,.08,.09]);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2.7);ax.set_title('Prespecified biological coherence: all seven datasets retained',pad=25);save(fig,'D5_FIG01_SEVEN_DATASET_COHERENCE_MATRIX.png')
 fig,ax=plt.subplots(figsize=(8,4.7));ax.axvline(0,color='.7',lw=.8);ax.scatter(p.cliffs_delta,yy,color='#0072B2');ax.set(yticks=yy,yticklabels=labels,xlim=(-1,1.1),xlabel="Primary Cliff's delta (positive minus call-negative)",title='Full-source primary effects; sparse/non-tests are not inferential');ax.invert_yaxis()
 for i,r in enumerate(p.itertuples()):
  if not np.isfinite(r.cliffs_delta):ax.text(.02,i,'Modality unavailable',va='center',color='.45')
  elif r.evaluability=='SPARSE_DESCRIPTIVE_ONLY':ax.annotate('Sparse: 2 positives',(r.cliffs_delta,i),xytext=(-120,-14),textcoords='offset points',fontsize=8)
 ax.set_ylim(6.7,-.7)
 save(fig,'D5_FIG02_PRIMARY_FEATURE_EFFECT_SIZE.png')
 fig,ax=plt.subplots(figsize=(8,4.7));valid=p.q.notna();ax.scatter(p.loc[valid,'q'],yy[valid],color='#0072B2');ax.set(yticks=yy,yticklabels=labels,xlabel='Within-family BH q (two-sided CRS association)',xlim=(0,max(.012,float(p.q.max())*1.25)),title='Primary FDR and direction; significance is not a QA gate');ax.invert_yaxis()
 for i,r in enumerate(p.itertuples()):
  ax.text(.0002,i-.13,'Higher CRS' if np.isfinite(r.delta_median) and r.delta_median>0 else 'Not evaluable' if not np.isfinite(r.delta_median) else 'Lower CRS',fontsize=8)
 ax.set_ylim(6.7,-.7);save(fig,'D5_FIG03_PRIMARY_FDR_AND_DIRECTION.png')
 fig,ax=plt.subplots(figsize=(8,4.7));ax.scatter(p.CRS_sign_consistent_fraction,yy-.12,label='Subset CRS',color='#0072B2');ax.scatter(p.OOF_sign_consistent_fraction,yy+.12,label='OOF ordinal',marker='s',color='#D55E00');ax.set(yticks=yy,yticklabels=labels,xlim=(0,1.18),xlabel='Same-sign fraction among evaluable subsets',title='B100 direction robustness (no iid inference)');ax.invert_yaxis();ax.legend(loc='lower left')
 for i,r in enumerate(p.itertuples()):
  if r.n_evaluable:ax.text(1.02,i,f'n={r.n_evaluable}',fontsize=8,va='center')
  else:ax.text(.03,i,'Not evaluable',color='.45',va='center')
 ax.set_ylim(6.7,-.7);ax.legend(loc='lower right');save(fig,'D5_FIG04_B100_DIRECTION_CONSISTENCY.png')
 for k,r in enumerate(cases.itertuples(),5):
  g=j[(j.cohort_id==r.cohort_id)&(j.feature==r.feature)&j.value.notna()].sort_values('cell_line');fig,ax=plt.subplots(figsize=(6,4.5))
  for value,color in [(0,'#7F7F7F'),(1,'#0072B2')]:
   v=g[g.value==value].CRS.to_numpy();offset=np.linspace(-.16,.16,len(v));ax.scatter(value+offset,v,s=22,color=color,alpha=.8);ax.plot([value-.2,value+.2],[np.median(v)]*2,color='black',lw=2)
  ax.set(xticks=[0,1],xticklabels=['No qualifying mutation call','Feature positive'],ylabel='Current D2 raw transported CRS',title=r.cohort_id+'\n'+r.feature)
  fig.text(.5,.01,f'{r.evaluability}; n+={r.n_positive}, n−={r.n_call_negative}; display-only selection',ha='center',fontsize=8);save(fig,f'D5_FIG{k:02d}_MECHANISTIC_CASE.png')
 sheet=Image.new('RGB',(1600,1400),'white')
 for i,path in enumerate(png):
  im=Image.open(path).convert('RGB');im.thumbnail((800,450));sheet.paste(im,((i%2)*800,(i//2)*460))
 sheet.save(ROOT/'work/d5_contact.png');shutil.copy2(__file__,OUT/'src/d5plots.py');print('Six PNGs generated')
if __name__=='__main__':main()
