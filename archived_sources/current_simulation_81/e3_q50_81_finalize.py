"""Presentation, report and verified delivery of the frozen 81-method readout."""
import csv,hashlib,json,shutil,time,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
import pymupdf
import ast,gzip,pickle,sys
from numpy.polynomial.legendre import leggauss

START=time.time()
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_RESULTS/TRIOS_PHASEC_E3_Q50_81METHOD_AUC_COMPLETION_v1.0_20260918'
PARENT=ROOT/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_3/DOWNSTREAM_COMPLETE_PIPELINE_v1.2a'
PRIOR=ROOT/'amendments/PhaseC_E3_Q75_81Method_Stage1_v2.1'
OLD=ROOT/'04_RESULTS/TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937/05_PHASE_C/E3'
S=json.loads((OUT/'FINAL_STATUS.json').read_text())
assert S['methods']==81 and S['old60_parity']=='PASS' and S['Q50_exact_log_TRIOS_parity']=='PASS'
R=pd.read_csv(OUT/'E3_81METHOD_RANKINGS.csv')
P=pd.read_csv(OUT/'E3_81METHOD_PARETO_FRONTIER.csv')
C=pd.read_csv(OUT/'E3_OLD61_VS_NEW81_HEADLINES.csv')
O=pd.read_csv(OUT/'E3_81METHOD_OPERATIONAL_SUMMARY.csv')
INF=pd.read_csv(OUT/'E3_81METHOD_CONFIRMATORY_INFERENCE_PARITY.csv')
sys.path.insert(0,str(ROOT/'04_RESULTS/HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE'))
from raw_models import MODEL_FUNCTIONS
scores=pd.read_csv(OUT/'E3_81METHOD_AUC_PROFILE_SCORES.csv',float_precision='round_trip')
bindings=[]
for p in sorted((PRIOR/'checkpoints').glob('*.pkl.gz')):
 with gzip.open(p,'rb') as f:q=pickle.load(f)
 bindings.append(dict(cohort_id=p.name[:-7],frozen_input_sha256=q['input_sha256'],checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),complete=q['complete']))
assert len(bindings)==2550 and all(x['complete'] for x in bindings)
pd.DataFrame(bindings).to_csv(OUT/'E3_81METHOD_FROZEN_INPUT_HASH_BINDINGS.csv',index=False)
nodes,weights=leggauss(1024);dose=125*(nodes+1)
qa_integral=[]
for p in sorted((ROOT/'work/trios_e3_downstream_v12a/inputs').glob('*.pkl.gz'))[::425]:
 with gzip.open(p,'rb') as f:inp=pickle.load(f)
 for rec in inp['parametric'][:4]:
  if not rec['fit_valid']:continue
  model=rec['model'];fun=MODEL_FUNCTIONS[model];theta=ast.literal_eval(rec['theta'])
  independent=float(np.sum(weights*fun(dose,theta))/2)
  production=float(scores.loc[(scores.sample_id==rec['sample_id'])&(scores.curve_model==model.replace('RAW_','')),'AUC_0_250'].iloc[0])
  delta=abs(independent-production)
  qa_integral.append(dict(input=p.name,sample_id=rec['sample_id'],model=model,production=production,independent_Gauss1024=independent,absolute_difference=delta,pass_tolerance=delta<1e-8))
assert len(qa_integral)>=20 and all(x['pass_tolerance'] for x in qa_integral)
pd.DataFrame(qa_integral).to_csv(OUT/'E3_81METHOD_INDEPENDENT_AUC_INTEGRATION_QA.csv',index=False,float_format='%.17g')
purple='#3c215d';teal='#0a817d';lav='#c7bdde';gray='#b0b7c1';blue='#667c91'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fd=OUT/'figures';fd.mkdir(exist_ok=True)
criteria=['ARI','BA','MacroF1','Extreme_LH','C_operational']
labels=['ARI','Balanced accuracy','Macro-F1','Extreme L-H error','Operational quality']
fig,ax=plt.subplots(1,5,figsize=(15,4),constrained_layout=True)
for j,c in enumerate(criteria):
 x=R[c].to_numpy();y=np.zeros(len(x));kind=R.representation.eq('AUC250');tri=R.method_id.eq('TRIOS')
 ax[j].scatter(x[~kind&~tri],y[~kind&~tri],c=gray,s=20,alpha=.7,label='Original conventional')
 ax[j].scatter(x[kind],y[kind],c=teal,s=25,marker='D',alpha=.8,label='AUC250')
 ax[j].scatter(x[tri],y[tri],c=purple,s=100,marker='*',label='Q50 TRIOS',zorder=5)
 ax[j].set_yticks([]);ax[j].set_title(labels[j],color=purple,fontsize=10)
 ax[j].grid(axis='x',alpha=.2)
fig.suptitle('E3 truth recovery and operational quality | 81 complete pipelines',color=purple,weight='bold')
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
fig.savefig(fd/'Figure_A_81Method_Landscape.pdf');fig.savefig(fd/'Figure_A_81Method_Landscape.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
tr=R[R.method_id=='TRIOS'].iloc[0]
ar=R[R.representation=='AUC250']
yy=np.arange(5)
auc_rank=[int(ar.sort_values(c,ascending=c=='Extreme_LH').iloc[0]['rank_'+c]) for c in criteria]
ax.scatter([int(tr['rank_'+c]) for c in criteria],yy,c=purple,s=100,marker='*',label='Q50 TRIOS',zorder=5)
ax.scatter(auc_rank,yy,c=teal,s=60,marker='D',label='Best AUC250 per criterion')
for i,r in enumerate(auc_rank):ax.plot([1,r],[i,i],color=lav,lw=2,zorder=0)
ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(.1,max(auc_rank)+3);ax.set_xlabel('Rank among 81 (lower is better)');ax.grid(axis='x',alpha=.2);ax.legend(frameon=False)
ax.set_title('Criterion-specific rank comparison',color=purple,weight='bold')
fig.savefig(fd/'Figure_B_Criterion_Rank_Strip.pdf');fig.savefig(fd/'Figure_B_Criterion_Rank_Strip.svg');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for k,(x,y) in enumerate((('ARI','BA'),('MacroF1','Extreme_LH'))):
 f=P.on_4D_frontier.astype(bool);a=P.representation.eq('AUC250');t=P.method_id.eq('TRIOS')
 ax[k].scatter(P.loc[~f,x],P.loc[~f,y],c=gray,s=15,alpha=.35,label='Dominated')
 ax[k].scatter(P.loc[f&~a&~t,x],P.loc[f&~a&~t,y],c=blue,s=48,label='Original frontier')
 ax[k].scatter(P.loc[f&a,x],P.loc[f&a,y],c=teal,s=55,marker='D',label='AUC250 frontier')
 ax[k].scatter(P.loc[t,x],P.loc[t,y],c=purple,s=120,marker='*',label='Q50 TRIOS',zorder=5)
 ax[k].set_xlabel(x);ax[k].set_ylabel(y+(' (lower better)' if y=='Extreme_LH' else ''));ax[k].grid(alpha=.2)
fig.suptitle('Four-dimensional Pareto membership, shown in paired projections',color=purple,weight='bold')
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
fig.savefig(fd/'Figure_C_4D_Pareto_Projection.pdf');fig.savefig(fd/'Figure_C_4D_Pareto_Projection.svg');plt.close(fig)

styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TRTitle',parent=styles['Title'],fontSize=21,leading=25,textColor=colors.HexColor(purple),spaceAfter=16));styles.add(ParagraphStyle(name='TRH',parent=styles['Heading2'],fontSize=12,leading=15,textColor=colors.HexColor(purple),spaceBefore=14));styles.add(ParagraphStyle(name='TRBody',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=7));styles.add(ParagraphStyle(name='TRSmall',parent=styles['BodyText'],fontSize=7.5,leading=11))
def para(s,style='TRBody'):return Paragraph(s,styles[style])
def table(rows,widths=None):
 t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(purple)),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f0f7')]),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#d9d5e2')),('FONTSIZE',(0,0),(-1,-1),7),('LEADING',(0,0),(-1,-1),10),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]));return t
story=[para('Phase C E3: Q50 81-method AUC completion','TRTitle'),para('Formal comparator-universe amendment | complete review required'),para('Reason and immutable parents','TRH'),para('The formal E3 benchmark previously compared Q50 / median TRIOS against 60 conventional pipelines. This amendment adds exactly 20 conventional AUC[0,250] pipelines, creating 81 complete pipelines. The frozen E3 truth bank contains 2,550 analysis sets, 53,100 profiles, and blocks B01-B05 only.'),para('The TRIOS arm is reused byte-for-byte from the current exact-log-space K=7 CRS / Admissible Breaks archive. Its transport, scores, labels and 61-method baseline are unchanged. No Q75 TRIOS quantity enters this result.'),para('AUC definition and reuse audit','TRH'),para('Each score is the fitted parametric response integrated from 0 to 250 micromolar, divided by 250; larger values indicate greater sensitivity. The four curves are Emax, Hill, Logistic and Weibull, crossed with Tertiles, Jenks, 1D K-means, 1D Ward and tied-variance three-component GMM.'),para('The prior Q75 Stage-1 archive supplied only the independent conventional AUC arm. All 2,550 input hashes matched sealed checkpoints. The source calls the original frozen E3 classifier and classifier-only LOO implementation. A total of 120 task-method combinations were independently replayed from frozen fits, including LOO, and matched the inherited output. All 153,000 original conventional task rows matched the frozen parent exactly. There were zero new curve fits, simulation runs, transport changes, or Q75 TRIOS imports.'),para('Factor balance and method-native validity','TRH'),para('All 81 methods use the frozen E3 factor weights across sample size / truth pattern, separation, generator family and paired block. Validity is method-native; failed tasks are retained in the applicability count. Operational quality uses the unchanged E3 conditional quality multiplied by factor-balanced applicability. Full classifier-only LOO records for the 20 AUC methods accompany this report.'),PageBreak(),para('Five criterion rankings','TRTitle')]
rows=[['Criterion','Q50 rank','Best conventional','Best AUC250','Q50 - best AUC']]
for r in C.itertuples():rows.append([r.criterion,str(r.new_trios_rank),r.new_best_conventional,r.best_AUC,f'{r.trios_minus_best_AUC:.6f}'])
story.extend([table(rows,[82,48,150,150,80]),Spacer(1,12),para('Four-dimensional truth Pareto','TRH'),para('Pareto dominance was recomputed jointly on ARI, balanced accuracy, Macro-F1 (higher better) and extreme L-H error (lower better), over all 81 methods. Operational quality was excluded from the frontier.'),para('Frontier methods: '+', '.join(S['pareto_frontier'])+'.'),para('Confirmatory inference parity','TRH'),para('The pre-specified Emax | EC50 | Tertiles comparator, 100,000 paired one-sided sign flips, four-endpoint Holm correction and frozen 5,000-draw paired block bootstrap are unchanged. No best-AUC significance test was added. The exact-log-space parent inference table is copied byte-for-byte.'),table([['Endpoint','Estimate','Raw P','Holm P']]+[[r.metric,f'{r.estimate:.6f}',f'{r.P_raw:.6g}',f'{r.P_Holm:.6g}'] for r in INF.itertuples()],[90,90,90,90]),PageBreak(),para('Old 61 versus new 81','TRTitle')])
rows=[['Criterion','61 rank','81 rank','Old best','New best','Headline']]
headline=[]
for r in C.itertuples():
 cls='PRESERVED' if r.old_trios_rank==r.new_trios_rank and r.old_best_conventional==r.new_best_conventional else ('WEAKENED_NUMERICALLY' if r.new_trios_rank>r.old_trios_rank else 'CHANGED')
 headline.append(dict(criterion=r.criterion,classification=cls));rows.append([r.criterion,str(r.old_trios_rank),str(r.new_trios_rank),r.old_best_conventional,r.new_best_conventional,cls])
story.extend([table(rows,[65,43,43,120,120,80]),Spacer(1,14),para('Limitations and interpretation','TRH'),para('This is the fixed formal E3 truth bank and these 81 complete pipelines only. Descriptive rank and Pareto conclusions are tied to the chosen AUC window, fit bank, and original factor-balanced E3 definition. The 20 added pipelines do not receive post-hoc confirmatory tests. The frozen parent bootstrap covers the original 61 methods; it is preserved as inference provenance, not misrepresented as an 81-method bootstrap.'),para('QA and reproducibility','TRH'),para('The delivery contains complete task records, 212,400 profile-model AUC scores, all method summaries, rankings, the full 4D frontier, new AUC classifier-only LOO folds, frozen parent inference and bootstrap tables, a source/seed-logic snapshot, parity audit, SHA-256 manifest and ZIP CRC check. Existing production, manuscript and presentation files were not modified.')])
report=OUT/'TRIOS_PhaseC_E3_Q50_81Method_AUC_Completion_Report_v1.0.pdf';SimpleDocTemplate(str(report),pagesize=(8.5*inch,11*inch),leftMargin=.6*inch,rightMargin=.6*inch,topMargin=.55*inch,bottomMargin=.5*inch).build(story)
S['headline_conclusions']=headline
S['report_pdf']=str(report)
S['wall_clock_seconds']=S['wall_clock_seconds']+time.time()-START
(OUT/'FINAL_STATUS.json').write_text(json.dumps(S,indent=2)+'\n')
source=OUT/'source_and_config';source.mkdir(exist_ok=True)
for src in [ROOT/'amendments/e3_q50_81_auc_completion.py',Path(__file__),PRIOR/'run_stage1.py',PRIOR/'source_binding.json',PARENT/'src/run_tasks.py',PARENT/'src/classifiers_v11_frozen.py',PARENT/'src/config_frozen.py',ROOT/'work/trios_e3_downstream_v12a/prepare.py',ROOT/'04_RESULTS/HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE/raw_models.py']:
 target=source/src.name
 if target.exists():target.chmod(0o666)
 shutil.copy2(src,target)
target=source/'FROZEN_Q50_PARENT_FINAL_STATUS.json'
if target.exists():target.chmod(0o666)
shutil.copy2(OLD/'FINAL_STATUS.json',target)
(source/'SEED_LOGIC.md').write_text('Frozen classifier seed derivation is in classifiers_v11_frozen.py (child_seed and deterministic start IDs). Formal E3 blocks are B01-B05; 2550 input hashes were checked against prior sealed checkpoints. No new seeds were drawn.\n')
manifest=[]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name not in ('MANIFEST.csv','SHA256SUMS.txt'):
  manifest.append((p.relative_to(OUT).as_posix(),p.stat().st_size,sha(p)))
with open(OUT/'MANIFEST.csv','w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(manifest)
(OUT/'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n,_,h in manifest))
zipname=ROOT/'TRIOS_PHASEC_E3_Q50_81METHOD_AUC_COMPLETION_v1.0_complete_delivery.zip'
with zipfile.ZipFile(zipname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
with zipfile.ZipFile(zipname) as z:
 assert z.testzip() is None
 for n,b,h in manifest:
  assert hashlib.sha256(z.read(n)).hexdigest()==h
S['complete_zip']=str(zipname);S['zip_sha256']=sha(zipname);S['crc_status']='PASS';S['payload_hash_verification']='PASS';S['wall_clock_seconds']=S['wall_clock_seconds']+time.time()-START
# Final status in ZIP records the scientific result; sidecar records delivery checks.
(OUT/'DELIVERY_VERIFICATION.json').write_text(json.dumps({k:S[k] for k in ('complete_zip','zip_sha256','crc_status','payload_hash_verification','wall_clock_seconds')},indent=2)+'\n')
doc=pymupdf.open(report)
for i,page in enumerate(doc):page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(str(OUT/f'report_page_{i+1:02d}.png'))
assert len(doc)>=3
print(json.dumps({k:S[k] for k in ('FINAL_STATUS','ranks','best_conventional','best_AUC','pareto_frontier','complete_zip','zip_sha256','crc_status','wall_clock_seconds')},indent=2))
