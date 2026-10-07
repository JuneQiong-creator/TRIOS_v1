"""Stage-1 E3 point estimates only. Reads certified fits; no fitting or inference."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import ast, gzip, hashlib, json, math, pickle, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad, IntegrationWarning
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
WORK=ROOT/'work/trios_e3_downstream_v12a'
PARENT=ROOT/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_3/DOWNSTREAM_COMPLETE_PIPELINE_v1.2a'
OLD=ROOT/'04_RESULTS/TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937/05_PHASE_C/E3'
sys.path.insert(0,str(WORK)); import prepare
sys.path.insert(0,str(PARENT/'src')); import run_tasks as rt
sys.path.insert(0,str(ROOT/'04_RESULTS/HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE'))
from raw_models import MODEL_FUNCTIONS
REC=prepare.load_recovery()
PN={12:3,18:12,24:19}
MODELS=('EMAX','HILL','LOGISTIC','WEIBULL'); ALGS=('TERTILES','JENKS','KMEANS','WARD','TIED_GMM')
MET=('ARI','BA','MacroF1','Extreme_LH'); COMP=('S_BW','S_tau','S_match','P_fit','J_SF')
CK=OUT/'checkpoints'; CK.mkdir(parents=True,exist_ok=True)
GRID=None

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 q=p.with_suffix('.tmp')
 with gzip.open(q,'wb') as f:pickle.dump(x,f,pickle.HIGHEST_PROTOCOL)
 os.replace(q,p)
def geometry(D):
 nodes=np.exp(np.linspace(0,math.log(D),7));nodes[0]=1.;nodes[-1]=D
 beta=np.arange(6,0,-1)*np.diff(nodes);beta=beta/beta.sum()
 return nodes,np.r_[beta[0]/2,(beta[:-1]+beta[1:])/2,beta[-1]/2]
def crs(preds,D):
 if D is None:return np.full(len(preds),np.nan)
 nodes,beta=geometry(D)
 return np.array([float(beta@p(nodes)) for p in preds])
def prestage(preds,grid):
 den=np.array([float(p([grid[-1]])[0]-p([0.])[0]) for p in preds]);info=np.isfinite(den)&(den>1e-12)
 au=[];tu=[]
 for u in grid:
  au.append(np.array([float(p([u])[0]-p([0.])[0]) for p in preds])/den)
  tu.append(np.array([float(abs(p([u])[0]-p([.6*u])[0])) for p in preds])/den)
 return info,np.array(au),np.array(tu)
def stage(preds,grid,omit=None,cache=None):
 n=len(preds)-(omit is not None);need=max(10,3*max(3,math.ceil(.15*n)))
 info,aa,tt=cache if cache is not None else prestage(preds,grid)
 info=info.copy()
 if omit is not None:info[omit]=False
 records=[];selected=None
 for j,u in enumerate(grid):
  au=aa[j];tu=tt[j]
  mask=info&np.isfinite(au)&np.isfinite(tu);m=int(mask.sum());req=math.ceil(.75*m)
  sa=int(np.sum(mask&(au>=.8)));st=int(np.sum(mask&(tu<=.1)));ok=m>=need and sa>=req and st>=req
  records.append((u,m,need,sa,st,req,ok))
  if ok and selected is None:selected=u
 return selected,records
def trios(d,preds,grid):
 cid=d['meta']['cohort_id'];ids=np.array(d['ids']);truth=np.array(d['truth']);n=len(ids)
 base={**d['meta'],'method_id':'TRIOS','curve_model':'SPLINE','representation':'CRS','stratifier':'TRIOS_ADMISSIBLE_BREAKS','method_family':'TRIOS','P_fit':d['P_fit']['SPLINE'],'n_profiles':n,'valid':False,'failure_code':'','LOO_successful_folds':0,'LOO_failed_folds':0}
 cache=prestage(preds,grid);D,candidates=stage(preds,grid,cache=cache);x=crs(preds,D);z=REC.ab(x,ids) if D is not None else None
 if z is None:return base|{'failure_code':'TRANSPORT_INCOMPATIBLE_NO_PASS' if D is None else 'NO_ADMISSIBLE_THREE_STRATUM_PARTITION'},[],candidates
 y,t1,t2=z;R=float(np.ptp(x));fold=[];ds=[];om=[];ret=[]
 for h,sid in enumerate(ids):
  keep=np.arange(n)!=h;Dh,_=stage(preds,grid,h,cache);xa=crs(preds,Dh);zz=REC.ab(xa[keep],ids[keep]) if Dh is not None else None;ok=zz is not None
  if ok:
   _,a,b=zz;yp=rt.held(xa,a,b,True);dd=.5*(((a-t1)/R)**2+((b-t2)/R)**2);ds.append(dd);om.append(float(yp[h]==y[h]));ret.append(float(np.mean(yp[keep]==y[keep])))
  fold.append(dict(cohort_id=cid,left_out_id=sid,fold_valid=ok,failure_code='' if ok else 'TRANSPORT_INCOMPATIBLE_NO_PASS' if Dh is None else 'NO_ADMISSIBLE_THREE_STRATUM_PARTITION',D_transport=Dh,tau1=a if ok else np.nan,tau2=b if ok else np.nan,d_h_squared=dd if ok else np.nan,omitted_match=om[-1] if ok else np.nan,retained_match=ret[-1] if ok else np.nan))
 A=len(ds)/n;stc=1/(1+math.sqrt(float(np.mean(ds)))) if ds else np.nan;ml=float(np.mean(om)) if om else 0.;mr=float(np.mean(ret)) if ret else 0.;smc=2*ml*mr/(ml+mr) if ml+mr else 0.;st=A*stc if ds else 0.;sm=A*smc
 mu=x.mean();wi=sum(float(np.sum((x[y==g]-x[y==g].mean())**2)) for g in 'LIH');be=sum(int(np.sum(y==g))*float((x[y==g].mean()-mu)**2) for g in 'LIH');sb=be/(be+wi);sizes={g:int(sum(y==g)) for g in 'LIH'};sf=float(min(sizes.values())>=2);ct=float(np.mean([sb,st,sm,base['P_fit'],sf]))
 tm=dict(ARI=float(adjusted_rand_score(truth,y)),BA=float(balanced_accuracy_score(truth,y)),MacroF1=float(f1_score(truth,y,labels=list('LIH'),average='macro',zero_division=0)),Extreme_LH=float(np.mean(((truth=='L')&(y=='H'))|((truth=='H')&(y=='L')))))
 return base|dict(valid=True,tau1=t1,tau2=t2,full_labels='|'.join(y),D_transport=D,group_n_L=sizes['L'],group_n_I=sizes['I'],group_n_H=sizes['H'],A_LOO=A,LOO_successful_folds=len(ds),LOO_failed_folds=n-len(ds),S_BW=sb,S_tau_conditional=stc,S_tau=st,S_match_conditional=smc,S_match=sm,J_SF=sf,C_task=ct,**tm),fold,candidates
def auc_methods(d):
 vals={};issues=[]
 for r in d['parametric']:
  m=r['model'];sid=r['sample_id']
  if not r['fit_valid']:vals[sid,m]=np.nan;continue
  theta=ast.literal_eval(r['theta']);fun=MODEL_FUNCTIONS[m]
  with warnings.catch_warnings(record=True) as seen:
   warnings.simplefilter('always',IntegrationWarning)
   v,err=quad(lambda x:float(fun([x],theta)[0]),0.,250.,epsabs=1e-10,epsrel=1e-10,limit=300)
  vals[sid,m]=v/250.
  for w in seen:issues.append(dict(cohort_id=d['meta']['cohort_id'],sample_id=sid,model=m,warning=str(w.message),error_estimate=err))
 rows=[];folds=[]
 for m in MODELS:
  transformed=[]
  for r in d['parametric']:
   q=r.copy()
   if q['model']=='RAW_'+m:q['FINITE_RANGE_EMAX']=vals[q['sample_id'],q['model']]
   transformed.append(q)
  dd=d.copy();dd['parametric']=transformed
  for a in ALGS:
   md=dict(method_id=f'{m}|AUC250|{a}',curve_model=m,representation='EMAX',stratifier=a,method_family='CONVENTIONAL_AUC250')
   row,ff,_=rt.evaluate(dd,md,[]);row['representation']='AUC250';rows.append(row);folds.extend(ff)
 return rows,folds,issues
def process(path):
 global GRID
 if GRID is None:GRID=json.loads((OUT/'measured_transport_candidates.json').read_text())
 d=pickle.load(gzip.open(path,'rb'));p=CK/(d['meta']['cohort_id']+'.pkl.gz')
 if p.exists():
  q=pickle.load(gzip.open(p,'rb'))
  if q.get('input_sha256')==sha(path) and q.get('complete'):return str(p)
 preds=[REC.pred_from_record(s) for s in d['curve_states']]
 tr,tf,cands=trios(d,preds,GRID);ar,af,issues=auc_methods(d)
 save(p,dict(input_sha256=sha(path),complete=True,rows=[tr]+ar,trios_folds=tf,auc_folds=af,candidates=cands,integration_warnings=issues))
 return str(p)
def aggregate(d,extra=()):
 rows=[]
 for key,g in d.groupby(list(extra)+['method_id'],sort=True):
  if not isinstance(key,tuple):key=(key,)
  w=g.fb_weight.to_numpy();v=g.valid.astype(bool).to_numpy();mass=float(w[v].sum());den=float(w.sum());r=dict(zip(list(extra)+['method_id'],key));r.update(intended_task_count=len(g),valid_task_count=int(v.sum()),A_valid=mass/den,full_data_failure_count=int((~v).sum()),LOO_failure_count=int(g.LOO_failed_folds.sum()),LOO_successful_count=int(g.LOO_successful_folds.sum()))
  for c in MET+COMP+('C_task','A_LOO'):r[c]=float(np.sum(w[v]*g.loc[v,c].to_numpy())/mass) if mass else np.nan
  r['C_conditional']=r.pop('C_task');r['C_operational']=r['C_conditional']*r['A_valid'];r['P_SF']=r['J_SF'];rows.append(r)
 return pd.DataFrame(rows)
def write(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,float_format='%.17g',lineterminator='\n');return p
def main():
 global GRID
 paths=sorted((WORK/'inputs').glob('*.pkl.gz'));assert len(paths)==2550
 sample=pickle.load(open(ROOT/'work/trios_phasec_sim_e1_v10/checkpoints_authoritative'/('E1_N12_P01_MODERATE_B01_EMAX.pkl'),'rb'))
 GRID=sorted({float(r['dose_uM']) for r in sample['raw'] if r['dose_uM']>0});assert len(GRID)==7 and GRID[0]==1. and GRID[-1]==250.
 (OUT/'measured_transport_candidates.json').write_text(json.dumps(GRID,indent=2)+'\n')
 assert len(pd.read_csv(PARENT/'01_TRIOS_PARITY/full_bank_parity.csv'))==2550
 assert len(pd.read_csv(PARENT/'02_MODEL_SUPPORT/profile_AICc_weights.csv.gz'))==265500
 print('Bound 2550 frozen inputs, 53100 certified profiles; candidates',GRID,flush=True)
 with ProcessPoolExecutor(max_workers=min(12,os.cpu_count() or 1)) as pool:
  futures={pool.submit(process,p):p for p in paths[:1550]}
  for i,f in enumerate(as_completed(futures),1):
   f.result()
   if i%100==0:print('Early cohort worker',i,'/1550',flush=True)
 while len(list(CK.glob('*.pkl.gz')))<2550:
  print('Waiting for disjoint late cohort worker; checkpoints',len(list(CK.glob('*.pkl.gz'))),'/2550',flush=True)
  time.sleep(30)
 print('Evaluation complete; aggregating',flush=True)
 recs=[pickle.load(gzip.open(p,'rb')) for p in sorted(CK.glob('*.pkl.gz'))];assert len(recs)==2550
 new=pd.DataFrame([r for q in recs for r in q['rows']]);assert len(new)==2550*21
 old=pd.read_csv(PARENT/'04_FULL_DATA/method_cohort_results.csv.gz',float_precision='round_trip');conv=old[old.method_id!='TRIOS'].copy();assert len(conv)==2550*60
 d=pd.concat([conv,new],ignore_index=True,sort=False);assert d.method_id.nunique()==81 and len(d)==2550*81
 d['fb_weight']=d.n.map(lambda n:1/(3*3*PN[n]*5*5));assert np.allclose(d.groupby('method_id').fb_weight.sum(),1,atol=1e-12,rtol=0)
 summary=aggregate(d).sort_values(['C_operational','method_id'],ascending=[False,True]);summary['rank_C_operational']=summary.C_operational.rank(ascending=False,method='min').astype(int)
 for c in MET:summary['rank_'+c]=summary[c].rank(ascending=c=='Extreme_LH',method='min').astype(int)
 tr=summary[summary.method_id=='TRIOS'].iloc[0];second=summary.iloc[1];gate=int(tr.rank_C_operational)==1
 write('TRIOS_PhaseC_E3_81Method_Coperational_Rank_v2.1.csv',summary)
 write('TRIOS_PhaseC_E3_AUC250_20_Methods_v2.1.csv',summary[summary.method_id.str.contains('AUC250')])
 write('TRIOS_PhaseC_E3_81Method_Truth_Diagnostics_v2.1.csv',summary[['method_id',*MET,*['rank_'+c for c in MET]]])
 robustness=pd.concat([aggregate(d,[c]).assign(factor=c) for c in ('n','separation','family')],ignore_index=True);write('TRIOS_PhaseC_E3_Robustness_Summaries_v2.1.csv',robustness)
 oldsum=pd.read_csv(OLD/'06_TRUTH_RECOVERY/method_level_61_complete_summary.csv');oldtr=oldsum[oldsum.method_id=='TRIOS'].iloc[0];cols=list(MET)+['C_operational','A_valid','A_LOO',*COMP];write('TRIOS_PhaseC_E3_Old_vs_Q75_v2.1.csv',[dict(metric=c,old=oldtr[c],Q75=tr[c],delta=tr[c]-oldtr[c]) for c in cols])
 write('evidence/TRIOS_cohort_results.csv.gz',new[new.method_id=='TRIOS']);write('evidence/AUC250_cohort_results.csv.gz',new[new.method_id!='TRIOS'])
 write('evidence/TRIOS_LOO.csv.gz',[f for q in recs for f in q['trios_folds']]);write('evidence/AUC250_LOO.csv.gz',[f for q in recs for f in q['auc_folds']]);write('evidence/integration_warnings.csv',[f for q in recs for f in q['integration_warnings']])
 folds=[f for q in recs for f in q['trios_folds']];vals=summary.set_index('method_id')[list(MET)].to_numpy()*[1,1,1,-1];mids=list(summary.method_id);ti=mids.index('TRIOS');front=not any(np.all(vals[j]>=vals[ti]) and np.any(vals[j]>vals[ti]) for j in range(81) if j!=ti);dom=sum(np.all(vals[ti]>=vals[j]) and np.any(vals[ti]>vals[j]) for j in range(81) if j!=ti)
 status=dict(FINAL_STATUS='PHASEC_E3_Q75_81METHOD_STAGE1_COP_RANK1_PASS_REVIEW_REQUIRED' if gate else 'PHASEC_E3_Q75_81METHOD_STAGE1_COP_GATE_REVIEW_REQUIRED',candidate_doses_uM=GRID,cohorts=2550,profiles=53100,methods=81,new_AUC250_methods=20,TRIOS_C_operational=float(tr.C_operational),TRIOS_rank=int(tr.rank_C_operational),rank2_method=second.method_id,rank2_C_operational=float(second.C_operational),rank2_gap=float(tr.C_operational-second.C_operational),TRIOS_A_valid=float(tr.A_valid),TRIOS_A_LOO=float(tr.A_LOO),TRIOS_failed_folds=int(sum(not f['fold_valid'] for f in folds)),TRIOS_truth={c:dict(value=float(tr[c]),rank=int(tr['rank_'+c])) for c in MET},Pareto_frontier=front,Pareto_dominated_methods=int(dom),new_optimizer_starts=0,bootstrap_runs=0,permutation_runs=0,Holm_runs=0,retuning=0)
 (OUT/'FINAL_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
 lines=['# E3 Stage-1 point-estimate readout','',f"**{status['FINAL_STATUS']}**",'',f"TRIOS C_operational {tr.C_operational:.12g}, rank {int(tr.rank_C_operational)}/81. Rank 2: {second.method_id}, {second.C_operational:.12g}; gap {tr.C_operational-second.C_operational:.12g}.",'',f'Candidates: {GRID}. Frozen 2550 cohorts / 53100 profiles. Q75 marginal gate, direct-log-space K7 CRS, current AB. 60 parent comparators reused; 20 AUC250 methods evaluated. Curve optimizer starts, bootstrap, permutation, Holm and retuning: zero.','']
 for c in MET:
  best=summary.sort_values([c,'method_id'],ascending=[c=='Extreme_LH',True]).iloc[0];lines.append(f"- {c}: TRIOS {tr[c]:.10g} (rank {int(tr['rank_'+c])}/81); best {best.method_id} {best[c]:.10g}.")
 (OUT/'TRIOS_PhaseC_E3_Q75_81Method_Stage1_v2.1_Report.md').write_text('\n'.join(lines)+'\n')
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.csv','SHA256SUMS.txt')];manifest=pd.DataFrame([dict(path=p.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]);write('manifest.csv',manifest);(OUT/'SHA256SUMS.txt').write_text(''.join(f'{r.sha256}  {r.path}\n' for r in manifest.itertuples()))
 print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
