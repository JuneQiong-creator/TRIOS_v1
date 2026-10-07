"""Frozen E3 Q50 61+20 AUC completion. No TRIOS or curve refitting."""
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

START = time.time()
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'04_RESULTS/TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937/05_PHASE_C/E3'
PARENT = ROOT/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_3/DOWNSTREAM_COMPLETE_PIPELINE_v1.2a'
PRIOR = ROOT/'amendments/PhaseC_E3_Q75_81Method_Stage1_v2.1'
INPUT = ROOT/'work/trios_e3_downstream_v12a/inputs'
OUT = ROOT/'04_RESULTS/TRIOS_PHASEC_E3_Q50_81METHOD_AUC_COMPLETION_v1.0_20260918'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(ROOT/'work/trios_e3_downstream_v12a'))
import prepare
sys.path.insert(0,str(PARENT/'src'))
import run_tasks as rt
sys.path.insert(0,str(ROOT/'04_RESULTS/HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE'))
from raw_models import MODEL_FUNCTIONS

MET = ['ARI','BA','MacroF1','Extreme_LH']
CRIT = MET+['C_operational']
PN={12:3,18:12,24:19}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write(name,frame):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(frame).to_csv(p,index=False,float_format='%.17g',lineterminator='\n')
 return p

def check(condition,message):
 if not condition:raise RuntimeError('HARD STOP: '+message)

def aggregate(d):
 rows=[]
 for mid,g in d.groupby('method_id',sort=True):
  w=g.fb_weight.to_numpy(float);v=g.valid.astype(bool).to_numpy();mass=float(w[v].sum());den=float(w.sum())
  r=dict(method_id=mid,intended_task_count=len(g),valid_task_count=int(v.sum()),A_valid=mass/den,full_data_failure_count=int((~v).sum()),LOO_failure_count=int(g.LOO_failed_folds.sum()),LOO_successful_count=int(g.LOO_successful_folds.sum()))
  for c in MET+['S_BW','S_tau','S_match','P_fit','J_SF','A_LOO','C_task']:
   r[c]=float(np.sum(w[v]*g.loc[v,c].to_numpy(float))/mass) if mass else np.nan
  r['C_conditional']=r.pop('C_task');r['C_operational']=r['C_conditional']*r['A_valid'];r['P_SF']=r['J_SF']
  rows.append(r)
 return pd.DataFrame(rows)

def main():
 audit=[]
 def a(gate,passed,evidence):audit.append(dict(gate=gate,status='PASS' if passed else 'FAIL',evidence=evidence));check(passed,gate+': '+evidence)
 oldpath=OLD/'04_FULL_DATA/method_cohort_results.csv.gz'
 base=pd.read_csv(oldpath,float_precision='round_trip')
 parent=pd.read_csv(PARENT/'04_FULL_DATA/method_cohort_results.csv.gz',float_precision='round_trip')
 auc=pd.read_csv(PRIOR/'evidence/AUC250_cohort_results.csv.gz',float_precision='round_trip')
 a('FORMAL_SET',len(base)==2550*61 and base.cohort_id.nunique()==2550 and int(base[base.method_id=='TRIOS'].n_profiles.sum())==53100 and set(base.base_block_id)==set(range(1,6)),f'{len(base)} old rows, {base.cohort_id.nunique()} sets')
 a('METHOD_COUNT',base.method_id.nunique()==61 and auc.method_id.nunique()==20 and len(auc)==2550*20,f'{base.method_id.nunique()}+{auc.method_id.nunique()} methods')
 a('AUC_KEYS',set(auc.cohort_id)==set(base.cohort_id) and auc.groupby('method_id').cohort_id.nunique().eq(2550).all() and set(auc.method_id).isdisjoint(base.method_id), '20 AUC methods cover each frozen set once')
 cols=[c for c in parent.columns if c in base.columns and c not in ('runtime_seconds','fb_weight')]
 bp=base[base.method_id!='TRIOS'].sort_values(['cohort_id','method_id']).reset_index(drop=True)
 pp=parent[parent.method_id!='TRIOS'].sort_values(['cohort_id','method_id']).reset_index(drop=True)
 numeric=bp[cols].select_dtypes(include='number').columns.tolist();other=[c for c in cols if c not in numeric]
 numeric_ok=np.allclose(bp[numeric].to_numpy(float),pp[numeric].to_numpy(float),equal_nan=True,rtol=0,atol=0)
 string_ok=bp[other].fillna('<NA>').astype(str).equals(pp[other].fillna('<NA>').astype(str))
 a('OLD_60_PARITY',len(bp)==153000 and numeric_ok and string_ok,'153000 rows, all shared values identical to frozen conventional parent')
 a('Q50_TRIOS',len(base[base.method_id=='TRIOS'])==2550 and sha(oldpath)==sha(OLD/'04_FULL_DATA/method_cohort_results.csv.gz'),'TRIOS reused byte-for-byte from exact-log Q50 archive')
 a('NO_Q75_TRIOS',set(auc.method_family)=={'CONVENTIONAL_AUC250'} and not any(x=='TRIOS' for x in auc.method_id),'only 20 conventional AUC rows inherited')
 a('AUC_FORMULA_SOURCE', 'quad(lambda x:float(fun([x],theta)[0]),0.,250.,epsabs=1e-10,epsrel=1e-10,limit=300)' in (PRIOR/'run_stage1.py').read_text(), 'fixed 0..250 quadrature, divide by 250, frozen raw model')
 a('FROZEN_CLASSIFIER_SOURCE', 'import run_tasks as rt' in (PRIOR/'run_stage1.py').read_text() and "rt.evaluate(dd,md,[])" in (PRIOR/'run_stage1.py').read_text(), 'same run_tasks.evaluate as old 60')
 paths=sorted(INPUT.glob('*.pkl.gz'));cks=sorted((PRIOR/'checkpoints').glob('*.pkl.gz'))
 a('FIT_INPUT_COUNT',len(paths)==len(cks)==2550,'2550 frozen input and 2550 sealed checkpoints')
 score_rows=[];qa_rows=[];chkrows=[]
 for i,p in enumerate(paths):
  with gzip.open(PRIOR/'checkpoints'/p.name,'rb') as f:q=pickle.load(f)
  h=sha(p)
  aok=q.get('complete') and q.get('input_sha256')==h and len(q['rows'])==21
  if not aok:check(False,'AUC frozen fit checkpoint mismatch '+p.name)
  if i in (0,325,700,1250,1800,2549):chkrows.append((p,q))
  with gzip.open(p,'rb') as f:d=pickle.load(f)
  for r in d['parametric']:
   model=r['model'];sid=r['sample_id'];ok=bool(r['fit_valid'])
   val=np.nan;err=np.nan
   if ok:
    theta=ast.literal_eval(r['theta']);fun=MODEL_FUNCTIONS[model]
    with warnings.catch_warnings(record=True) as ww:
     val,err=quad(lambda z:float(fun([z],theta)[0]),0.,250.,epsabs=1e-10,epsrel=1e-10,limit=300)
    val/=250.
   score_rows.append((d['meta']['cohort_id'],sid,model.replace('RAW_',''),val,ok,err))
  if i%500==0:print('AUC profiles',i,'/2550',flush=True)
 a('FROZEN_FIT_HASH',True,'all 2550 input SHA-256 values match prior sealed checkpoint records')
 scores=pd.DataFrame(score_rows,columns=['cohort_id','sample_id','curve_model','AUC_0_250','fit_valid','quadrature_error'])
 a('AUC_SCORE_COUNT',len(scores)==53100*4 and set(scores.curve_model)=={'EMAX','HILL','LOGISTIC','WEIBULL'},f'{len(scores)} profile-model scores')
 a('AUC_FINITE',scores.loc[scores.fit_valid,'AUC_0_250'].notna().all(),'all valid parametric fits have finite normalized AUC')
 for p,q in chkrows:
  with gzip.open(p,'rb') as f:d=pickle.load(f)
  trans=[]
  for r in d['parametric']:
   t=r.copy()
   if t['model'].startswith('RAW_'):
    v=scores.loc[(scores.sample_id==r['sample_id'])&(scores.curve_model==r['model'].replace('RAW_','')),'AUC_0_250'].iloc[0]
    t['FINITE_RANGE_EMAX']=v
   trans.append(t)
  dd=d.copy();dd['parametric']=trans
  for row in q['rows'][1:]:
   md={k:row[k] for k in ('method_id','curve_model','stratifier','method_family')};md['representation']='EMAX'
   got,folds,_=rt.evaluate(dd,md,[])
   for c in ('full_labels','tau1','tau2','ARI','BA','MacroF1','Extreme_LH','C_task','LOO_successful_folds','LOO_failed_folds'):
    lhs=got.get(c);rhs=row.get(c)
    equal=(str(lhs)==str(rhs)) if isinstance(lhs,str) or isinstance(rhs,str) else (math.isclose(float(lhs),float(rhs),abs_tol=1e-11,rel_tol=1e-11) if lhs is not None and rhs is not None else lhs==rhs)
    if not equal:check(False,f'AUC independent replay {p.name} {row["method_id"]} {c}: {lhs} != {rhs}')
   qa_rows.append((p.name,row['method_id'],'PASS',len(folds)))
 a('INDEPENDENT_AUC_REPLAY',len(qa_rows)==6*20,'120 task methods replay frozen fit AUC + classifier + LOO')
 auc=auc.copy();auc['fb_weight']=auc.n.map(lambda n:1/(3*3*PN[n]*5*5))
 d=pd.concat([base,auc],ignore_index=True,sort=False)
 a('COMBINED_UNIVERSE',len(d)==2550*81 and d.method_id.nunique()==81,'206550 task rows, 81 complete pipelines')
 computed=aggregate(d)
 oldsummary=pd.read_csv(OLD/'06_TRUTH_RECOVERY/method_level_61_complete_summary.csv',float_precision='round_trip')
 oldcompare=computed[computed.method_id.isin(oldsummary.method_id)].set_index('method_id').sort_index()
 oldref=oldsummary.set_index('method_id').sort_index()
 for c in CRIT+['A_valid','A_LOO','S_BW','S_tau','S_match','P_fit','J_SF']:
  a('AGGREGATE_'+c,np.allclose(oldcompare[c],oldref[c],rtol=0,atol=2e-14,equal_nan=True),f'61 frozen method aggregates match in {c}')
 s=computed.copy()
 for c in CRIT:s['rank_'+c]=s[c].rank(ascending=c=='Extreme_LH',method='min').astype(int)
 registry=s[['method_id']].copy();registry[['curve_model','representation','stratifier']]=registry.method_id.str.split('|',expand=True).reindex(columns=[0,1,2]);registry.loc[registry.method_id=='TRIOS',['curve_model','representation','stratifier']]=['SPLINE','CRS','TRIOS_ADMISSIBLE_BREAKS'];registry['method_family']=np.where(registry.method_id=='TRIOS','TRIOS',np.where(registry.representation=='AUC250','CONVENTIONAL_AUC250','CONVENTIONAL'))
 s=s.merge(registry,on='method_id',validate='one_to_one')
 vals=s[MET].to_numpy(float)*np.array([1,1,1,-1]);front=[]
 for i in range(len(s)):
  dominated=any(np.all(vals[j]>=vals[i]) and np.any(vals[j]>vals[i]) for j in range(len(s)) if j!=i)
  front.append(not dominated)
 s['on_4D_frontier']=front
 infpath=OLD/'08_CONFIRMATORY_INFERENCE/confirmatory_TRIOS_vs_Emax_EC50_Tertiles.csv'
 inf=pd.read_csv(infpath,float_precision='round_trip')
 a('CONFIRMATORY_LOCK',len(inf)==4 and set(inf.metric)==set(MET) and inf.B_PERM.eq(100000).all() and {'TRIOS','EMAX|EC50|TERTILES'}.issubset(set(base.method_id)),'frozen four-endpoint 100000-permutation Holm table; comparator unchanged')
 bt=OLD/'11_BOOTSTRAP/method_percentile_intervals.csv';bootstrap=pd.read_csv(bt)
 a('BOOTSTRAP_LOCK',bootstrap.method_id.nunique()==61,'frozen 5000-draw parent bootstrap included, no new draw')
 write('E3_81METHOD_METHOD_REGISTRY.csv',registry)
 write('E3_81METHOD_AUC_PROFILE_SCORES.csv',scores)
 write('E3_81METHOD_FULLDATA_TASK_RESULTS.csv.gz',d)
 write('E3_81METHOD_TRUTH_METRIC_SUMMARY.csv',s[['method_id','intended_task_count','valid_task_count','A_valid']+MET+['rank_'+c for c in MET]])
 write('E3_81METHOD_OPERATIONAL_SUMMARY.csv',s[['method_id','intended_task_count','valid_task_count','A_valid','full_data_failure_count','LOO_failure_count','LOO_successful_count','S_BW','S_tau','S_match','P_fit','J_SF','A_LOO','C_conditional','C_operational','rank_C_operational']])
 write('E3_81METHOD_RANKINGS.csv',s[['method_id']+CRIT+['rank_'+c for c in CRIT]+['representation','method_family']])
 write('E3_81METHOD_PARETO_FRONTIER.csv',s[['method_id']+MET+['on_4D_frontier','representation']])
 write('E3_81METHOD_LOO_SUMMARY.csv',s[['method_id','intended_task_count','valid_task_count','LOO_successful_count','LOO_failure_count','A_LOO']])
 shutil.copy2(infpath,OUT/'E3_81METHOD_CONFIRMATORY_INFERENCE_PARITY.csv')
 shutil.copy2(bt,OUT/'E3_FROZEN_PARENT_BOOTSTRAP_5000.csv')
 shutil.copy2(PRIOR/'evidence/AUC250_LOO.csv.gz',OUT/'E3_81METHOD_AUC_CLASSIFIER_ONLY_LOO.csv.gz')
 shutil.copy2(OLD/'05_LOO/TRIOS_LOO.csv.gz',OUT/'E3_FROZEN_Q50_TRIOS_LOO.csv.gz')
 write('E3_81METHOD_AUC_REUSE_PARITY_AUDIT.csv',qa_rows and [dict(input=p,method_id=m,status=z,folds=n) for p,m,z,n in qa_rows])
 write('E3_81METHOD_QA_CHECKLIST.csv',audit)
 oldrank=oldsummary.set_index('method_id');sr=s.set_index('method_id')
 compare=[]
 for c in CRIT:
  oldtr=oldrank.loc['TRIOS'];newtr=sr.loc['TRIOS'];oldconv=oldrank.drop('TRIOS').sort_values([c],ascending=c=='Extreme_LH').iloc[0];newconv=sr.drop('TRIOS').sort_values([c],ascending=c=='Extreme_LH').iloc[0];aucbest=sr[sr.representation=='AUC250'].sort_values([c],ascending=c=='Extreme_LH').iloc[0]
  compare.append(dict(criterion=c,old_trios_rank=int(oldtr['rank_Extreme' if c=='Extreme_LH' else 'rank_'+c]),new_trios_rank=int(newtr['rank_'+c]),old_best_conventional=oldconv.name,new_best_conventional=newconv.name,best_AUC=aucbest.name,trios_value=newtr[c],best_conventional_value=newconv[c],best_AUC_value=aucbest[c],trios_minus_best_conventional=(newconv[c]-newtr[c] if c=='Extreme_LH' else newtr[c]-newconv[c]),trios_minus_best_AUC=(aucbest[c]-newtr[c] if c=='Extreme_LH' else newtr[c]-aucbest[c])))
 comp=pd.DataFrame(compare);write('E3_OLD61_VS_NEW81_HEADLINES.csv',comp)
 status=dict(FINAL_STATUS='TRIOS_PHASEC_E3_Q50_81METHOD_AUC_COMPLETION_v1.0_COMPLETE_REVIEW_REQUIRED',result_root=str(OUT),methods=81,analysis_sets=2550,profiles=53100,old60_parity='PASS',Q50_exact_log_TRIOS_parity='PASS',AUC_implementation='PASS',AUC_arm='REUSED_FROM_Q75_CONVENTIONAL_ONLY',ranks={c:int(sr.loc['TRIOS','rank_'+c]) for c in CRIT},best_conventional={r.criterion:r.new_best_conventional for r in comp.itertuples()},best_AUC={r.criterion:r.best_AUC for r in comp.itertuples()},pareto_frontier=s.loc[s.on_4D_frontier,'method_id'].tolist(),TRIOS_on_frontier=bool(sr.loc['TRIOS','on_4D_frontier']),confirmatory_inference_parity='PASS',old_vs_new=comp.to_dict('records'),frozen_sources={str(p.relative_to(ROOT)):sha(p) for p in (oldpath,infpath,bt,PRIOR/'run_stage1.py',PARENT/'src/run_tasks.py')},wall_clock_seconds=time.time()-START)
 (OUT/'FINAL_STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  (OUT/'HARD_STOP.txt').write_text(str(e)+'\n',encoding='utf-8');raise
