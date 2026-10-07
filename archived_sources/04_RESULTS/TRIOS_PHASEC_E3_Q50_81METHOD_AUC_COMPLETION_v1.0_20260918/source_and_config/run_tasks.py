"""E3 frozen downstream tasks. No curve fitting, inversions, or optimization."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import concurrent.futures as cf,gzip,hashlib,json,math,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score,f1_score
from prepare import ROOT,P,OUT,WORK,load_recovery
SRC=ROOT/'04_RESULTS/HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE'
CL=ROOT/'04_RESULTS/UPLOAD_BUNDLES_FINAL/TRIOS_FINAL_UPLOAD_HB_PDO_v1.0/02_FINAL_SCRIPTS/final_method'
sys.path[:0]=[str(CL),str(SRC)]
import classifiers_v11_frozen as clf
REC=load_recovery()
MODELS=('EMAX','HILL','LOGISTIC','WEIBULL');REPS=('IC50','EC50','EMAX');ALGS=('TERTILES','JENKS','KMEANS','WARD','TIED_GMM')
METHODS=[dict(method_id='TRIOS',curve_model='SPLINE',representation='CRS',stratifier='TRIOS_ADMISSIBLE_BREAKS',method_family='TRIOS')]+[dict(method_id=f'{m}|{r}|{a}',curve_model=m,representation=r,stratifier=a,method_family='CONVENTIONAL') for m in MODELS for r in REPS for a in ALGS]
CK=WORK/'checkpoints';CK.mkdir(exist_ok=True)
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 with gzip.open(tmp,'wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
 os.replace(tmp,path)
def held(x,a,b,trios):return np.where(x<=a,'L',np.where(x<=b if trios else x<b,'I','H'))
def partition(x,ids,md,tag):
 if not np.isfinite(x).all():return dict(valid=False,failure_code='NONFINITE_REQUIRED_PROFILE_SCORE')
 if md['method_id']=='TRIOS':
  z=REC.ab(x,ids)
  return dict(valid=False,failure_code='NO_ADMISSIBLE_THREE_STRATUM_PARTITION') if z is None else dict(valid=True,labels=z[0],tau1=z[1],tau2=z[2])
 return clf.fit_classifier(x,ids,md['stratifier'],tag)[0]
def checked_valid(z,x,trios):
 if not z.get('valid'):return False
 assert np.isfinite([z['tau1'],z['tau2']]).all() and z['tau1']<z['tau2']
 assert len(set(z['labels']))==3
 assert np.array_equal(held(x,z['tau1'],z['tau2'],trios),z['labels'])
 return True
def evaluate(d,md,preds):
 cid=d['meta']['cohort_id'];ids=np.array(d['ids']);truth=np.array(d['truth']);n=len(ids);mid=md['method_id'];trios=mid=='TRIOS'
 base={**d['meta'],**md,'valid':False,'failure_code':'','P_fit':d['P_fit'][md['curve_model']],'n_profiles':n,'LOO_successful_folds':0,'LOO_failed_folds':0}
 rows=[];folds=[]
 if trios:
  D=d['TRIOS_endpoint'];x=np.array(d['TRIOS_scores']);p=d['TRIOS_parent']
  if p is None or not p['valid']:return base|dict(failure_code='TRANSPORT_INCOMPATIBLE_NO_PASS' if D is None else 'NO_ADMISSIBLE_THREE_STRATUM_PARTITION'),folds,rows
  full=dict(valid=True,tau1=p['tau1'],tau2=p['tau2'],labels=np.array([d['TRIOS_labels'][s] for s in ids]))
 else:
  D=None;by={r['sample_id']:r for r in d['parametric'] if r['model']=='RAW_'+md['curve_model']};col={'IC50':'IC50','EC50':'EC50','EMAX':'FINITE_RANGE_EMAX'}[md['representation']]
  x=np.array([by[s][col] if by[s]['fit_valid'] else np.nan for s in ids],float)
  if md['representation']!='EMAX':x=-x
  rows=[dict(cohort_id=cid,method_id=mid,sample_id=s,oriented_score=float(v),fit_valid=bool(by[s]['fit_valid']),representation_status=by[s].get(col+'_failure',''),extrapolated=bool(by[s].get('IC50_extrapolated',False)) if col=='IC50' else False) for s,v in zip(ids,x)]
  if not np.isfinite(x).all():
   code='TRUE_CURVE_FIT_FAILURE' if any(not by[s]['fit_valid'] for s in ids) else ('NO_FINITE_IC50_CROSSING' if col=='IC50' else 'NONFINITE_REQUIRED_PROFILE_SCORE')
   return base|dict(failure_code=code),folds,rows
  full=partition(x,ids,md,f'B3FINAL|{mid}|{cid}|FULL')
 if not checked_valid(full,x,trios):return base|dict(failure_code=full.get('failure_code','STRATIFIER_FAILURE')),folds,rows
 y=np.array(full['labels']);R=float(np.ptp(x));assert R>0
 if trios:rows=[dict(cohort_id=cid,method_id=mid,sample_id=s,oriented_score=float(v),D_transport=D) for s,v in zip(ids,x)]
 for h,sid in enumerate(ids):
  keep=np.arange(n)!=h;Dh=None
  if trios:
   Dh,_=REC.transport([p for k,p in enumerate(preds) if k!=h]);xa=REC.crs(preds,Dh) if Dh is not None else np.full(n,np.nan)
  else:xa=x
  z=partition(xa[keep],ids[keep],md,f'B3FINAL|{mid}|{cid}|LOO|{sid}');ok=checked_valid(z,xa[keep],trios)
  pred=held(xa,z['tau1'],z['tau2'],trios) if ok else np.full(n,'FAIL')
  dh=.5*(((z['tau1']-full['tau1'])/R)**2+((z['tau2']-full['tau2'])/R)**2) if ok else np.nan
  folds.append(dict(cohort_id=cid,method_id=mid,left_out_id=sid,fold_valid=ok,failure_code='' if ok else ('TRANSPORT_INCOMPATIBLE_NO_PASS' if trios and Dh is None else z.get('failure_code','LOO_FOLD_FAILURE')),tau1=z.get('tau1',np.nan),tau2=z.get('tau2',np.nan),D_transport=Dh,d_h_squared=dh,omitted_match=float(pred[h]==y[h]) if ok else np.nan,retained_match=float(np.mean(pred[keep]==y[keep])) if ok else np.nan,heldout_label=str(pred[h]),fold_labels='|'.join(pred),fold_scores='|'.join(format(v,'.17g') for v in xa) if trios else '',selected_start_id=z.get('selected_start_id',None)))
 good=[f for f in folds if f['fold_valid']];A=len(good)/n
 if good:
  stc=1/(1+math.sqrt(float(np.mean([f['d_h_squared'] for f in good]))));ml=np.mean([f['omitted_match'] for f in good]);mr=np.mean([f['retained_match'] for f in good]);smc=float(2*ml*mr/(ml+mr)) if ml+mr else 0.;st=A*stc;sm=A*smc
 else:stc=smc=np.nan;st=sm=0.
 mu=float(np.mean(x));wi=sum(float(np.sum((x[y==g]-np.mean(x[y==g]))**2)) for g in 'LIH');be=sum(int(np.sum(y==g))*float((np.mean(x[y==g])-mu)**2) for g in 'LIH');sbw=be/(be+wi)
 sizes={g:int(np.sum(y==g)) for g in 'LIH'};sf=float(min(sizes.values())>=2);pfit=base['P_fit'];ct=float(np.mean([sbw,st,sm,pfit,sf]));assert np.isfinite(ct)
 truthmet=dict(ARI=float(adjusted_rand_score(truth,y)),BA=float(balanced_accuracy_score(truth,y)),MacroF1=float(f1_score(truth,y,labels=list('LIH'),average='macro',zero_division=0)),Extreme_LH=float(np.mean(((truth=='L')&(y=='H'))|((truth=='H')&(y=='L')))))
 if trios:
  for out,old in [('ARI','ARI'),('BA','balanced_accuracy'),('MacroF1','macro_f1'),('Extreme_LH','extreme_error')]:assert truthmet[out]==d['TRIOS_parent'][old],('TRIOS_TRUTH_PARITY',cid,out)
 return base|dict(valid=True,tau1=full['tau1'],tau2=full['tau2'],full_labels='|'.join(y),D_transport=D,group_n_L=sizes['L'],group_n_I=sizes['I'],group_n_H=sizes['H'],A_LOO=A,LOO_successful_folds=len(good),LOO_failed_folds=n-len(good),S_BW=sbw,S_tau_conditional=stc,S_tau=st,S_match_conditional=smc,S_match=sm,J_SF=sf,C_task=ct,selected_start_id=full.get('selected_start_id',None),**truthmet),folds,rows
def cohort_run(path):
 d=pickle.load(gzip.open(path,'rb'));cid=d['meta']['cohort_id'];target=CK/(cid+'.pkl.gz')
 input_sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 ans=pickle.load(gzip.open(target,'rb')) if target.exists() else dict(cohort_id=cid,tasks=[],folds=[],scores=[],complete=False,input_sha256=input_sha)
 assert ans['input_sha256']==input_sha
 done={r['method_id'] for r in ans['tasks']};preds=[REC.pred_from_record(s) for s in d['curve_states']]
 for md in METHODS:
  if md['method_id'] in done:continue
  t=time.time();row,folds,scores=evaluate(d,md,preds);row['runtime_seconds']=time.time()-t
  ans['tasks'].append(row);ans['folds'].extend(folds);ans['scores'].extend(scores);ans['last_completed_method']=md['method_id'];ans['updated_at']=time.time();ans['complete']=len(ans['tasks'])==61;save(target,ans)
 return cid,len(ans['tasks'])
def main():
 assert json.loads((WORK/'PREPARED.json').read_text())['status']=='PASS';assert os.cpu_count()>=16
 (OUT/'01_METHOD_REGISTRY').mkdir(exist_ok=True);pd.DataFrame(METHODS).to_csv(OUT/'01_METHOD_REGISTRY/method_registry.csv',index=False)
 paths=sorted((WORK/'inputs').glob('*.pkl.gz'));assert len(paths)==2550
 print(f'[{time.strftime("%F %T")}] DOWNSTREAM RUN: 16 workers x 1 thread; 2550 cohorts x 61 methods; no curve refits',flush=True)
 pool=cf.ProcessPoolExecutor(max_workers=16);todo=iter(paths);futs={pool.submit(cohort_run,next(todo)) for _ in range(16)};j=0
 try:
  while futs:
   done,_=cf.wait(futs,timeout=30,return_when=cf.FIRST_COMPLETED)
   if not done:print(f'[{time.strftime("%F %T")}] RUNNING {len(futs)} active cohorts; completed {j}/2550',flush=True)
   for f in done:
    futs.remove(f);cid,count=f.result();j+=1;print(f'[{time.strftime("%F %T")}] COMPLETE COHORT {j}/2550 {cid}; {count}/61 methods checkpointed',flush=True)
    nxt=next(todo,None)
    if nxt is not None:futs.add(pool.submit(cohort_run,nxt))
 except BaseException:
  (WORK/'EXECUTION_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
  for process in pool._processes.values():process.terminate()
  pool.shutdown(wait=True,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 (WORK/'TASKS_COMPLETE.json').write_text(json.dumps(dict(status='TASKS_COMPLETE_PENDING_QA',cohorts=2550,methods=61)),encoding='utf-8')
if __name__=='__main__':main()
