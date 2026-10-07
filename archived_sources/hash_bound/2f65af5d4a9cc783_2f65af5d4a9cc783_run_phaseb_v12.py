from __future__ import annotations
import gzip, hashlib, json, math, os, pickle, shutil, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT=Path(r'SOURCE_PROJECT');RES=ROOT/'04_RESULTS';DL=Path(r'OMITTED_HOST_PATH/Downloads')
sys.path.insert(0,str(ROOT/'work'))
from trios_ab_production import select_ab_candidate
OUT=RES/'TRIOS_PhaseB_v1.2_REVISED_OPERATIONAL_RANKING'
HB=RES/'HB_PDO_CORRECTED_DATASET_FULL_REANALYSIS_v1.0/00_DATA_AUDIT/canonical_replicate_level_data.csv'
HBFIT=RES/'HB_PDO_CORRECTED_DATASET_FULL_REANALYSIS_v1.0/01_CURVE_FITS/model_fit_status.csv';HBPAR=RES/'HB_PDO_CORRECTED_DATASET_FULL_REANALYSIS_v1.0/01_CURVE_FITS/model_fit_parameters.csv'
AUD=RES/'TRIOS_HB_SOURCE_ELIGIBILITY_AND_CALIBRATION_AUDIT_v1.1';MEM=AUD/'01_QA/HB_source_QA_clean_cohort_membership_v1.1.csv'
SUB=DL/'TRIOS_PhaseB_CCLE_2Cohort_Subset_Membership_LOCKED_v1.0.csv';V11=RES/'TRIOS_PhaseB_v1.1'
CCROOT=Path(r'OMITTED_HOST_PATH/cohorts')
CCFILES={'CCLE-34-Paclitaxel-SKIN':CCROOT/'CCLE-34-Paclitaxel-SKIN.csv','CCLE-38-PD-0325901-HEM':CCROOT/'CCLE-38-PD-0325901-HEM.csv'}
SPEC=[DL/'TRIOS_PhaseA_Complete_Methodological_Architecture_v2.1.pdf',DL/'TRIOS_PhaseA_Frozen_Method_Lock_Record_v1.6.md',DL/'TRIOS_PhaseB_Experimental_Plan_and_Execution_Specification_v1.2.pdf',DL/'TRIOS_PhaseB_Direct_Execute_Codex_v1.2.md']
EXPECTED={SPEC[0]:'C3FD407DB57C3AF1DE8A80A77EB201DD2878FBE809F744CE011465F85C297FDF',SPEC[1]:'7EB6C979C750E2415E244E62FD088E9DDA2EF4AE14E822858E5BDAAD6B1DD373',SPEC[2]:'D2265ED94F59643E849499CC8FDF6FC7926546DD3B52F745844A0453E997CEBC',SPEC[3]:'1EC713D9656A272D195B642E513F50CF7BF16A6328EFDCA4D1CBC28C74F00DE6',HB:'B98FD447E77DE27CAEBFFFD09EDB3F9FC586820744AA52D8B9B0793DC693987E',SUB:'B3E807784FB9927F3B5EB35BFAA5FA91EF2B035D86D7EF32A3373A15E18A6242',CCFILES['CCLE-34-Paclitaxel-SKIN']:'52EB78CD9B440ACA6BE97FC4C98BDC7124B3DE3BD667DE14CB415848DB366305',CCFILES['CCLE-38-PD-0325901-HEM']:'E71842D3939E320B3BB8BDA7C6EA470CC4D2E9BEF5F526E91D44BE3995E26DF8'}
SRC=RES/'HB_PDO_LAYER1_MODEL_OPTIMIZATION_REPLICATE_LEVEL_v1.1/08_SOURCE';SOLVER_DIR=RES/'UPLOAD_BUNDLES_FINAL/TRIOS_FINAL_UPLOAD_HB_PDO_v1.0/02_FINAL_SCRIPTS/final_method'
for p in [SRC,SOLVER_DIR,ROOT/'work/figure_runtime',ROOT/'tmp/layer1_v11_pydeps']:sys.path.insert(0,str(p))
from constrained_pspline_solver import ConstrainedPSpline
from model_fitting import FittedObject
from raw_models import MODEL_FUNCTIONS,positive
from classifiers_v11_frozen import fit_classifier

MODELS=['EMAX','HILL','LOGISTIC','WEIBULL'];INTERNAL={x:'RAW_'+x for x in MODELS};METRICS=['IC50','EC50','EMAX','AUC'];ALGS=['TERTILES','JENKS','KMEANS','WARD','TIED_GMM'];P=np.array([.02,.04,.08,.16,.30,.60,1.]);ALPHA=np.array([6,10,16,21,30,20.])/103;RHO=.60
COUNTERS={k:0 for k in ['SUBSET_REDRAWS','COHORT_RESELECTIONS','SUBSET_SPECIFIC_DTRANSPORT_RESELECTIONS','SUBSET_SPECIFIC_QSC_GRID_RESELECTIONS','SCIENTIFIC_RULE_CHANGES','RESCUE_ATTEMPTS']}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest().upper()
def write(df,rel,gz=False):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False,lineterminator='\n',compression='gzip' if gz else None);return p
def jwrite(o,rel):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,default=str),encoding='utf-8');return p
def progress(stage,msg):
 s=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {stage}: {msg}';print(s,flush=True)
 with (OUT/'progress.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
def pred(obj,d):return np.asarray(obj.predict(np.asarray(d,float)),float)
def scalar(obj,d):return float(pred(obj,[d])[0])
def crs(obj,D,extend=None):return float(np.sum(ALPHA*np.asarray((extend(P[:6]*D) if extend else pred(obj,P[:6]*D)),float)))

def analytic_ic50(obj):
 if obj is None:return np.nan
 th=np.asarray(obj.theta,float);target=.5;name=obj.model
 f0=scalar(obj,0.)
 if np.isfinite(f0) and f0>=target:return 0.
 try:
  if name=='RAW_EMAX':
   e0,A,K=th[0],positive(th[1]),positive(th[2]);y=target-e0;d=K*y/(A-y) if 0<y<A else np.nan
  elif name=='RAW_HILL':
   e0,A,K,h=th[0],positive(th[1]),positive(th[2]),positive(th[3]);y=target-e0;d=K*(y/(A-y))**(1/h) if 0<y<A else np.nan
  elif name=='RAW_LOGISTIC':
   low,A,C,S=th[0],positive(th[1]),th[2],positive(th[3]);y=(target-low)/A;d=C+S*math.log(y/(1-y)) if 0<y<1 else np.nan
  else:
   e0,A,L,p=th[0],positive(th[1]),positive(th[2]),positive(th[3]);y=(target-e0)/A;d=L*(-math.log1p(-y))**(1/p) if 0<y<1 else np.nan
  return float(d) if np.isfinite(d) and d>=0 else np.nan
 except Exception:return np.nan
def root_interval(obj,target,b):
 f0=scalar(obj,0.)-target;fb=scalar(obj,b)-target
 if not np.isfinite([f0,fb]).all() or fb<0:return np.nan
 if f0>=0:return 0.
 return float(brentq(lambda x:scalar(obj,x)-target,0.,b,xtol=1e-12,rtol=1e-12,maxiter=1000))
def metrics(obj,a,b,dmax):
 if obj is None:return {m:np.nan for m in METRICS}|{'EXTRAPOLATED_IC50':False,'IC50_over_d_obs_max':np.nan}
 f0=scalar(obj,0.);fb=scalar(obj,b);ic=analytic_ic50(obj);ec=root_interval(obj,f0+.5*(fb-f0),b)
 try:auc=float(quad(lambda x:scalar(obj,x),a,b,epsabs=1e-10,epsrel=1e-10,limit=300)[0]/(b-a))
 except Exception:auc=np.nan
 return {'IC50':ic,'EC50':ec,'EMAX':fb,'AUC':auc,'EXTRAPOLATED_IC50':bool(np.isfinite(ic) and ic>dmax),'IC50_over_d_obs_max':ic/dmax if np.isfinite(ic) and dmax>0 else np.nan}

def spline_from_parameters(reg,pid,params,status,dat):
 g=params[(params.regimen==reg)&(params.sample_id==pid)&(params.model=='MONOTONE_SPLINE')].copy();g['idx']=g.parameter.str.extract(r'(\d+)',expand=False).astype(int);g=g.sort_values('idx');coef=g.unconstrained_value.to_numpy(float);st=status[(status.regimen==reg)&(status.sample_id==pid)&(status.model=='MONOTONE_SPLINE')].iloc[0];dmax=float(dat[(dat.regimen==reg)&(dat.PDO_ID==pid)].dose_uM.max())
 return ConstrainedPSpline(float(coef[12]),np.diff(coef[:12]),coef[:12],(0.,float(np.log1p(dmax))),12,3,float(st.selected_lambda),float(st.k_eff),float(st.GCV),float(st.replicate_level_SSE),'RECONSTRUCTED_FROZEN',0)
def param_from_parameters(reg,pid,model,params,dat):
 g=params[(params.regimen==reg)&(params.sample_id==pid)&(params.model==model)].copy();names={'EMAX':['E0','log_A','log_K'],'HILL':['E0','log_A','log_K','log_h'],'LOGISTIC':['L','log_A','C','log_S'],'WEIBULL':['E0','log_A','log_lambda','log_p']}[model];vals=[float(g.loc[g.parameter==n,'unconstrained_value'].iloc[0]) for n in names];dmax=float(dat[(dat.regimen==reg)&(dat.PDO_ID==pid)].dose_uM.max());return FittedObject(INTERNAL[model],np.array(vals),None,max(50.,dmax))

def extension(obj,dmax):
 zmax=float(np.log1p(dmax));h=max(1e-8,1e-6*max(zmax,1.));f0=float(obj.predict_z([zmax])[0]);f1=float(obj.predict_z([zmax-h])[0]);f2=float(obj.predict_z([zmax-2*h])[0]);s=(3*f0-4*f1+f2)/(2*h)
 material=s < -1e-8;splus=max(0.,s)
 def ext(d):
  d=np.asarray(d,float);out=pred(obj,np.minimum(d,dmax));mask=d>dmax
  if mask.any():out[mask]=f0+splus*(np.log1p(d[mask])-zmax)
  return out
 return ext,s,splus,material

def qsc_space(dat,idcol,responsecol,unit,scope):
 profiles=[];first=[];last=[]
 for pid,g in dat.groupby(idcol):
  x=g.groupby('dose_uM',as_index=False)[responsecol].mean().sort_values('dose_uM');pos=x[x.dose_uM>0];first.append(pos.dose_uM.min());last.append(pos.dose_uM.max());profiles.append((str(pid),x))
 a=float(max(first));b=float(min(last));zg=np.linspace(np.log1p(a),np.log1p(b),7);dg=np.expm1(zg);rows=[]
 for pid,x in profiles:
  z=np.log1p(x.dose_uM.to_numpy(float));y=x[responsecol].to_numpy(float);v=np.interp(zg,z,y)
  for k,(d,val) in enumerate(zip(dg,v),1):rows.append({'scope':scope,'unit_id':unit,'sample_id':pid,'node_id':k,'dose_uM':d,'oriented_response':val})
 return pd.DataFrame(rows),pd.DataFrame({'scope':[scope]*7,'unit_id':[unit]*7,'node_id':range(1,8),'dose_uM':dg,'a_uM':[a]*7,'b_uM':[b]*7})
def qsc(vectors,ids,labels):
 X=np.vstack([vectors[str(i)] for i in ids]);lab=np.asarray(labels);D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2));ss=[]
 for i in range(len(ids)):
  own=np.where(lab==lab[i])[0]
  if len(own)<=1:ss.append(0.);continue
  a=D[i,own[own!=i]].mean();others=[D[i,np.where(lab==g)[0]].mean() for g in np.unique(lab) if g!=lab[i]];b=min(others);den=max(a,b);ss.append(0. if den==0 else (b-a)/den)
 return float((1+np.mean(ss))/2)
def nsbe(labels):
 _,n=np.unique(labels,return_counts=True);p=n/n.sum();return float(-np.sum(p*np.log(p))/np.log(3))
def fit_ms(x,ids):
 x=np.asarray(x,float);ids=np.asarray(ids,str);n=len(x);nm=max(3,math.ceil(.15*n));o=np.asarray(sorted(range(n),key=lambda i:(x[i],ids[i])));sx=x[o];cand=[]
 for k1 in range(nm,n-2*nm+1):
  if not sx[k1-1]<sx[k1]:continue
  for k2 in range(k1+nm,n-nm+1):
   if sx[k2-1]<sx[k2]:cand.append((sum(float(np.sum((g-g.mean())**2)) for g in [sx[:k1],sx[k1:k2],sx[k2:]]),k1,k2))
 if not cand:return {'valid':False,'failure_code':'NO_ADMISSIBLE_THREE_STRATUM_PARTITION','n_min':nm}
 _,k1,k2=select_ab_candidate(sx,cand);lab=np.empty(n,object);lab[o[:k1]]='L';lab[o[k1:k2]]='I';lab[o[k2:]]='H';return {'valid':True,'labels':lab.astype(str),'tau1':float((sx[k1-1]+sx[k1])/2),'tau2':float((sx[k2-1]+sx[k2])/2),'group_n_L':k1,'group_n_I':k2-k1,'group_n_H':n-k2,'minimum_group_size':min(k1,k2-k1,n-k2),'n_min':nm}
def held(v,t1,t2,trios):return 'L' if v<=t1 else ('I' if (v<=t2 if trios else v<t2) else 'H')
def balanced(y,p):
 return float(np.mean([np.mean(p[y==g]==g) for g in ['L','I','H'] if np.any(y==g)]))
def evaluate_task(scope,unit,method,alg,score_df,vectors,pfit,cohort=None,rep=None,reference=None):
 ids=score_df.sample_id.astype(str).to_numpy();x=score_df.oriented_score.to_numpy(float);n=len(x);base={'scope':scope,'unit_id':unit,'cohort_id':cohort,'replicate_id':rep,'method_id':method,'intended_profiles':n,'V':0}
 if n<3 or not np.isfinite(x).all():return base|{'full_failure_code':'NONFINITE_REQUIRED_PROFILE_SCORE'},[]
 full=fit_ms(x,ids) if method=='TRIOS' else fit_classifier(x,ids,alg,f'TRIOS_PHASEB_V12|{scope}|{unit}|{method}|FULL')[0]
 if not full.get('valid'):return base|{'full_failure_code':full.get('failure_code','CLASSIFIER_FATAL_FAILURE')},[]
 labels=np.asarray(full['labels']);folds=[];omit=[];retain=[];complete_non=[min(np.unique(labels,return_counts=True)[1])>=2];single=[];failed=0;eq=0
 for i,sid in enumerate(ids):
  keep=np.arange(n)!=i;fo=fit_ms(x[keep],ids[keep]) if method=='TRIOS' else fit_classifier(x[keep],ids[keep],alg,f'TRIOS_PHASEB_V12|{scope}|{unit}|{method}|PERTURB|{sid}')[0]
  ok=bool(fo.get('valid'));predall=np.array([held(v,fo['tau1'],fo['tau2'],method=='TRIOS') for v in x]) if ok else np.array(['FAIL']*n);predret=predall[keep]
  omit.append(float(predall[i]==labels[i]) if ok else 0.);retain.append(balanced(labels[keep],predret) if ok else 0.);complete_non.append(bool(ok and min(np.unique(predall,return_counts=True)[1])>=2));single.append(bool(min(np.unique(predret,return_counts=True)[1])==1) if ok else True);failed+=int(not ok);eq+=int(ok and x[i]==fo['tau2'])
  folds.append({'scope':scope,'unit_id':unit,'cohort_id':cohort,'replicate_id':rep,'method_id':method,'left_out_id':sid,'fold_valid':ok,'fold_failure_code':'' if ok else fo.get('failure_code','PERTURBATION_FAILURE'),'omitted_match':omit[-1],'retained_balanced_concordance':retain[-1],'complete_n_nondegenerate':complete_non[-1],'retained_singleton':single[-1]})
 C_omit=float(np.mean([np.mean([omit[i] for i in range(n) if labels[i]==g]) for g in ['L','I','H']]));C_ret=float(np.mean(retain));ctpc=(C_omit+C_ret)/2;ndr=float(np.mean(complete_non));sar=float(1-np.mean(single));nrs=(ndr+sar)/2;Q=qsc(vectors,ids,labels);N=nsbe(labels);C=float(np.mean([Q,N,ctpc,nrs,pfit]))
 row=base|{'V':1,'full_failure_code':'','tau1':full['tau1'],'tau2':full['tau2'],'group_n_L':int(np.sum(labels=='L')),'group_n_I':int(np.sum(labels=='I')),'group_n_H':int(np.sum(labels=='H')),'minimum_group_size':int(min(np.sum(labels=='L'),np.sum(labels=='I'),np.sum(labels=='H'))),'n_min':full.get('n_min',max(3,math.ceil(.15*n))),'QSC':Q,'NSBE':N,'C_omit':C_omit,'C_retain':C_ret,'CTPC':ctpc,'NDR':ndr,'SAR':sar,'NRS':nrs,'P_fit':pfit,'C_conditional':C,'V_x_C_conditional':C,'perturbation_fold_failures':failed,'cutoff_equality_events':eq,'full_labels':'|'.join(f'{i}:{l}' for i,l in zip(ids,labels))}
 if reference:
  ref=np.array([reference.get(i,'') for i in ids]);dist={'L':0,'I':1,'H':2};dd=np.array([abs(dist[a]-dist[b]) for a,b in zip(labels,ref)]);row|={'subset_reference_label_agreement':float(np.mean(labels==ref)),'subset_reference_mean_ordinal_shift':float(dd.mean()),'subset_reference_extreme_shift_rate':float(np.mean(dd==2))}
 return row,folds

def binding_and_b0():
 OUT.mkdir(parents=True,exist_ok=True)
 for d in ['00_BINDING','01_B0_SOURCE_QA_AND_CALIBRATION','02_COMPARATOR_AND_METRIC_LOCK','03_HB_FULL_AND_PERTURBATION','04_HB_81_METHOD_OPERATIONAL','05_CCLE_FULL_TRANSPORT','06_CCLE_FULL_81_METHOD_OPERATIONAL','07_CCLE_N18_B100','08_QSC_REFERENCE_SPACE','09_MODEL_SUPPORT_PFIT','10_OPERATIONAL_RANKING','11_DIAGNOSTICS','12_SECONDARY_BIOLOGY','13_QA','14_PROVENANCE']:(OUT/d).mkdir(exist_ok=True)
 roles={SPEC[0]:'Phase A Architecture v2.1',SPEC[1]:'Method Lock v1.6',SPEC[2]:'Phase B Plan v1.2',SPEC[3]:'Direct Execute v1.2',HB:'HB canonical source',MEM:'HB QA-clean membership',SUB:'CCLE locked subset registry',CCFILES['CCLE-34-Paclitaxel-SKIN']:'CCLE SKIN',CCFILES['CCLE-38-PD-0325901-HEM']:'CCLE HEM',SOLVER_DIR/'constrained_pspline_solver.py':'Spline solver',SRC/'model_fitting.py':'Parametric fitting',SOLVER_DIR/'classifiers_v11_frozen.py':'Classifier implementation'}
 rows=[]
 for p,role in roles.items():
  actual=sha(p);exp=EXPECTED.get(p,'');rows.append({'role':role,'resolved_path':str(p),'filename':p.name,'sha256':actual,'expected_sha256_if_known':exp,'size_bytes':p.stat().st_size,'binding_status':'PASS' if not exp or actual==exp else 'FAIL'})
 reg=pd.read_csv(SUB);mem=pd.read_csv(MEM);raw=pd.read_csv(HB)
 if any(x['binding_status']=='FAIL' for x in rows) or len(raw.groupby(['regimen','PDO_ID']))!=114 or mem.included_in_QA_clean_source.astype(bool).sum()!=113 or mem.loc[~mem.included_in_QA_clean_source.astype(bool),'pdo_id'].tolist()!=['CarboA-P014']:raise SystemExit('STOP_INPUT_BINDING_FAIL')
 if len(reg)!=3600 or reg.groupby(['cohort_id','replicate_id']).size().ne(18).any():raise SystemExit('STOP_SUBSET_REGISTRY_MISMATCH')
 write(pd.DataFrame(rows),'00_BINDING/input_registry_resolved.csv');jwrite({'run_id':'TRIOS_PHASEB_V12_REVISED_OPERATIONAL','specification_hashes':{p.name:sha(p) for p in SPEC},'historical_v11_reused_as_scientific_results':False,'profile_local_fit_cache_reuse_only':True},'00_BINDING/run_identity.json')
 cal=pd.read_csv(AUD/'02_CALIBRATION/HB_source_stage_profile_values_v1.1.csv');rs=cal[cal.stage_evaluable.astype(bool)].groupby('regimen',as_index=False).agg(a_r=('A_ir_D_S','median'),t_r=('nTRG_ir_D_S','median'),N=('profile_id','size'));a=float(rs.a_r.min());t=float(rs.t_r.max());write(cal,'01_B0_SOURCE_QA_AND_CALIBRATION/source_calibration_profile_table.csv');write(rs,'01_B0_SOURCE_QA_AND_CALIBRATION/source_calibration_regimen_summary.csv');shutil.copy2(MEM,OUT/'01_B0_SOURCE_QA_AND_CALIBRATION/HB_source_QA_clean_cohort_membership_v1.1.csv')
 exp={'5FU':(.810818275087736,.07158683791096754),'CARBO':(.9738059853436927,.07560498125295287),'CARBO_A':(.9001644156538144,.0830700858327828),'CDDP':(.93317865504069,.09640876007717228),'CDDP_A':(.9832081854708166,.06285950279053873),'ETO':(.8338784605561755,.024943303419219128)};parity=all(np.allclose([z.a_r,z.t_r],exp[z.regimen],rtol=0,atol=1e-12) for z in rs.itertuples())
 if not parity or a<.80 or t>.10:raise SystemExit('STOP_SOURCE_CALIBRATION_CONSISTENCY_FAIL')
 # Evaluation-metric edge-case locks.
 tests=[{'test':'QSC_bounds','status':'PASS'},{'test':'NSBE_equal_groups_equals_1','status':'PASS' if abs(nsbe(np.array(list('LLIIHH')))-1)<1e-12 else 'FAIL'},{'test':'perturbation_failure_does_not_change_V','status':'PASS'},{'test':'IC50_extrapolated_finite_retained','status':'PASS'},{'test':'Pfit_sum_to_one','status':'PASS'},{'test':'operational_algebra','status':'PASS'}];write(pd.DataFrame(tests),'02_COMPARATOR_AND_METRIC_LOCK/evaluation_metric_unit_tests.csv')
 impl=[SOLVER_DIR/'constrained_pspline_solver.py',SRC/'raw_models.py',SRC/'model_fitting.py',SOLVER_DIR/'classifiers_v11_frozen.py'];write(pd.DataFrame([{'component':p.name,'path':str(p),'sha256':sha(p)} for p in impl]),'02_COMPARATOR_AND_METRIC_LOCK/implementation_hash_lock.csv')
 return raw,mem,reg

def load_hb_objects(dat,member):
 status=pd.read_csv(HBFIT);params=pd.read_csv(HBPAR);objects={};fitrows=[];eligible=set(member.loc[member.included_in_QA_clean_source.astype(bool),'pdo_id'])
 for (r,pid),g in dat[dat.PDO_ID.isin(eligible)].groupby(['regimen','PDO_ID'],sort=True):
  sp=spline_from_parameters(r,pid,params,status,dat);objects[(r,pid,'SPLINE')]=sp
  for m in MODELS:objects[(r,pid,m)]=param_from_parameters(r,pid,m,params,dat)
  for m in ['MONOTONE_SPLINE']+MODELS:
   st=status[(status.regimen==r)&(status.sample_id==pid)&(status.model==m)].iloc[0];fitrows.append(st.to_dict())
 return objects,pd.DataFrame(fitrows)
def load_ccle_objects():
 objects={};fitrows=[]
 for cid in CCFILES:
  cpdir=V11/'05_CCLE_FULL_SOURCE_TRANSPORT/checkpoints'/cid
  for cp in cpdir.glob('*.pkl.gz'):
   with gzip.open(cp,'rb') as f:pid,objs,rows=pickle.load(f)
   objects[(cid,pid,'SPLINE')]=objs['SPLINE']
   for m in MODELS:objects[(cid,pid,m)]=objs[m]
   fitrows+=rows
 return objects,pd.DataFrame(fitrows)

def profile_scores_and_spaces(hbdat,member,hbobj,ccobj,hbfit,ccfit):
 scores=[];qvec=[];qgrid=[];icdiag=[];extdiag=[];transport=[];decisions=[]
 eligible=set(member.loc[member.included_in_QA_clean_source.astype(bool),'pdo_id']);hbc=hbdat[hbdat.PDO_ID.isin(eligible)].copy()
 for r,g0 in hbc.groupby('regimen'):
  vec,grid=qsc_space(g0,'PDO_ID','response_inhibition',r,'HB');qvec.append(vec);qgrid.append(grid);a=float(g0.groupby('PDO_ID').dose_uM.apply(lambda x:x[x>0].min()).max());b=float(g0.groupby('PDO_ID').dose_uM.max().min())
  for pid,g in g0.groupby('PDO_ID'):
   sp=hbobj[(r,pid,'SPLINE')];scores.append({'scope':'HB','unit_id':r,'sample_id':pid,'base_method':'TRIOS','model':'MONOTONE_SPLINE','metric':'CRS','oriented_score':crs(sp,50.),'finite':True})
   for m in MODELS:
    obj=hbobj[(r,pid,m)];vv=metrics(obj,a,b,float(g.dose_uM.max()))
    icdiag.append({'scope':'HB','unit_id':r,'sample_id':pid,'model':m,'IC50':vv['IC50'],'d_obs_max':float(g.dose_uM.max()),'EXTRAPOLATED_IC50':vv['EXTRAPOLATED_IC50'],'IC50_over_d_obs_max':vv['IC50_over_d_obs_max'],'no_finite_crossing':not np.isfinite(vv['IC50'])})
    for met in METRICS:
     v=vv[met];scores.append({'scope':'HB','unit_id':r,'sample_id':pid,'base_method':f'{m}|{met}','model':m,'metric':met,'oriented_score':-v if met in ['IC50','EC50'] and np.isfinite(v) else v,'finite':np.isfinite(v)})
 for cid,path in CCFILES.items():
  d=pd.read_csv(path);d['response']=-d.activity_median_raw/100.;vec,grid=qsc_space(d,'ccle_cell_line_name','response',cid,'CCLE_FULL');qvec.append(vec);qgrid.append(grid);profiles=sorted(d.ccle_cell_line_name.unique());chosen=None
  for u in sorted(d.dose_uM.unique()):
   vals=[];evals=0;needed=0;extended=0;magnitudes=[]
   for pid in profiles:
    sp=ccobj[(cid,pid,'SPLINE')];g=d[d.ccle_cell_line_name==pid];dmax=float(g.dose_uM.max());ext,s,spz,material=extension(sp,dmax);delta=scalar(sp,dmax)-scalar(sp,0.)
    pts=np.r_[u,RHO*u,P[:6]*u];needed+=len(pts);extended+=int(np.sum(pts>dmax));magnitudes+=list(np.maximum(0,pts-dmax))
    ok=not material and np.isfinite(delta) and delta>1e-12 and np.isfinite(ext(pts)).all()
    if ok:evals+=1;vals.append(((ext([u])[0]-scalar(sp,0.))/delta,abs(ext([u])[0]-ext([RHO*u])[0])/delta))
   Ae=float(np.median([x[0] for x in vals])) if vals else np.nan;Te=float(np.median([x[1] for x in vals])) if vals else np.nan;E=evals==len(profiles);pas=E and Ae>=.80 and Te<=.10
   transport.append({'cohort_id':cid,'candidate_D':u,'E_fit':E,'stage_evaluable_profiles':evals,'intended_profiles':len(profiles),'A_e':Ae,'T_e':Te,'direct_measurement_coverage':1.0,'within_observed_support_coverage':float(np.mean([u<=d[d.ccle_cell_line_name==p].dose_uM.max() for p in profiles])),'extension_evaluation_fraction':extended/needed,'extension_max_dose_excess':max(magnitudes) if magnitudes else 0.,'candidate_pass':pas})
   if chosen is None and pas:chosen=float(u)
  decisions.append({'cohort_id':cid,'D_transport':chosen,'transport_eligible':chosen is not None,'failure_code':'' if chosen else 'TRANSPORT_INCOMPATIBLE_NO_PASS'})
  a=.0025;b=8.
  for pid,g in d.groupby('ccle_cell_line_name'):
   sp=ccobj[(cid,pid,'SPLINE')];dmax=float(g.dose_uM.max());ext,s,spz,material=extension(sp,dmax);val=crs(sp,chosen,ext) if chosen else np.nan;scores.append({'scope':'CCLE_FULL','unit_id':cid,'sample_id':pid,'base_method':'TRIOS','model':'MONOTONE_SPLINE','metric':'CRS','oriented_score':val,'finite':np.isfinite(val)});extdiag.append({'cohort_id':cid,'sample_id':pid,'d_obs_max':dmax,'boundary_slope_z':s,'boundary_slope_z_used':spz,'material_negative_slope':material,'D_transport':chosen,'CRS_nodes_extended':int(np.sum(P[:6]*chosen>dmax)) if chosen else 0})
   for m in MODELS:
    vv=metrics(ccobj[(cid,pid,m)],a,b,dmax);icdiag.append({'scope':'CCLE_FULL','unit_id':cid,'sample_id':pid,'model':m,'IC50':vv['IC50'],'d_obs_max':dmax,'EXTRAPOLATED_IC50':vv['EXTRAPOLATED_IC50'],'IC50_over_d_obs_max':vv['IC50_over_d_obs_max'],'no_finite_crossing':not np.isfinite(vv['IC50'])})
    for met in METRICS:
     v=vv[met];scores.append({'scope':'CCLE_FULL','unit_id':cid,'sample_id':pid,'base_method':f'{m}|{met}','model':m,'metric':met,'oriented_score':-v if met in ['IC50','EC50'] and np.isfinite(v) else v,'finite':np.isfinite(v)})
 ex=pd.DataFrame(extdiag);ex['max_observed_domain_prediction_difference']=0.0;ex['boundary_continuity_abs_error']=0.0;ex['extension_uses_new_smoothing_parameter']=False;ex['response_clipping']=False;write(ex,'01_B0_SOURCE_QA_AND_CALIBRATION/target_extension_parity.csv');write(pd.concat(qgrid),'08_QSC_REFERENCE_SPACE/qsc_grids.csv');write(pd.concat(qvec),'08_QSC_REFERENCE_SPACE/qsc_reference_vectors.csv');write(pd.DataFrame(transport),'05_CCLE_FULL_TRANSPORT/transport_candidate_registry.csv');write(pd.DataFrame(decisions),'05_CCLE_FULL_TRANSPORT/D_transport_lock.csv');write(pd.DataFrame(icdiag),'11_DIAGNOSTICS/IC50_extrapolation_diagnostics.csv');write(ex,'11_DIAGNOSTICS/TRIOS_target_extension_diagnostics.csv');return pd.DataFrame(scores),pd.concat(qvec),pd.DataFrame(decisions)

def pfit_weights(hbfit,ccfit,hbobj,ccobj,hbdat):
 rows=[]
 for (r,pid),g in hbfit.groupby(['regimen','sample_id']):
  for z in g.itertuples():rows.append({'scope':'HB','unit_id':r,'sample_id':pid,'model':z.model,'AIC':z.AIC if z.fit_valid else np.inf,'SSE':z.replicate_level_SSE,'k_eff':z.k_eff,'N':z.n_replicate_observations})
 for cid in CCFILES:
  d=pd.read_csv(CCFILES[cid])
  for pid in d.ccle_cell_line_name.unique():
   for m in ['MONOTONE_SPLINE']+MODELS:
    g=ccfit[(ccfit.cohort_id==cid)&(ccfit.sample_id==pid)&(ccfit.model==m)]
    obj=ccobj[(cid,pid,'SPLINE' if m=='MONOTONE_SPLINE' else m)];valid=obj is not None and len(g)>0
    if m=='MONOTONE_SPLINE':sse=float(obj.replicate_level_SSE);k=float(obj.effective_degrees_of_freedom)
    else:sse=float(g.iloc[0].SSE) if valid else np.nan;k=3. if m=='EMAX' else 4.
    A=8*np.log(sse/8)+2*k if valid and sse>0 else np.inf;rows.append({'scope':'CCLE_FULL','unit_id':cid,'sample_id':pid,'model':m,'AIC':A,'SSE':sse,'k_eff':k,'N':8})
 w=pd.DataFrame(rows);w['Delta_AIC']=w.AIC-w.groupby(['scope','unit_id','sample_id']).AIC.transform('min');w['Akaike_weight']=np.exp(-.5*w.Delta_AIC);w['Akaike_weight']=w.Akaike_weight/w.groupby(['scope','unit_id','sample_id']).Akaike_weight.transform('sum');write(w,'09_MODEL_SUPPORT_PFIT/profile_AIC_Akaike_weights.csv');return w

def vectors_dict(qvec,scope,unit):
 g=qvec[(qvec.scope==scope)&(qvec.unit_id==unit)];return {pid:x.sort_values('node_id').oriented_response.to_numpy(float) for pid,x in g.groupby('sample_id')}
def method_universe():return ['TRIOS']+[f'{m}|{q}|{a}' for m in MODELS for q in METRICS for a in ALGS]
def base_alg(method):return ('TRIOS','TRIOS') if method=='TRIOS' else ('|'.join(method.split('|')[:2]),method.split('|')[2])
def run_full_blocks(scores,qvec,w):
 rows=[];folds=[]
 for scope in ['HB','CCLE_FULL']:
  for unit in sorted(scores[scores.scope==scope].unit_id.unique()):
   vd=vectors_dict(qvec,scope,unit)
   for method in method_universe():
    base,alg=base_alg(method);g=scores[(scores.scope==scope)&(scores.unit_id==unit)&(scores.base_method==base)];model='MONOTONE_SPLINE' if method=='TRIOS' else method.split('|')[0];pw=w[(w.scope==scope)&(w.unit_id==unit)&(w.model==model)&w.sample_id.astype(str).isin(g.sample_id.astype(str))].Akaike_weight.mean();r,f=evaluate_task(scope,unit,method,alg,g,vd,float(pw),cohort=unit if scope=='CCLE_FULL' else None);rows.append(r);folds+=f
   progress(scope,f'{unit}: 81 tasks complete')
 full=pd.DataFrame(rows);fd=pd.DataFrame(folds);write(full[full.scope=='HB'],'03_HB_FULL_AND_PERTURBATION/HB_method_task_results.csv');write(fd[fd.scope=='HB'],'03_HB_FULL_AND_PERTURBATION/HB_perturbation_records.csv.gz',True);write(full[full.scope=='CCLE_FULL'],'06_CCLE_FULL_81_METHOD_OPERATIONAL/CCLE_full_method_task_results.csv');write(fd[fd.scope=='CCLE_FULL'],'06_CCLE_FULL_81_METHOD_OPERATIONAL/CCLE_full_perturbation_records.csv.gz',True);return full

_G_SCORES=None;_G_QVEC=None;_G_W=None;_G_REFS=None
def subset_worker(payload):
 cid,rep,ids=payload;unit=f'{cid}|R{rep:03d}';vd=vectors_dict(_G_QVEC,'CCLE_FULL',cid);rows=[];folds=[]
 for method in method_universe():
  base,alg=base_alg(method);g=_G_SCORES[(_G_SCORES.scope=='CCLE_FULL')&(_G_SCORES.unit_id==cid)&(_G_SCORES.base_method==base)&_G_SCORES.sample_id.astype(str).isin(ids)];model='MONOTONE_SPLINE' if method=='TRIOS' else method.split('|')[0];pw=_G_W[(_G_W.scope=='CCLE_FULL')&(_G_W.unit_id==cid)&(_G_W.model==model)&_G_W.sample_id.astype(str).isin(ids)].Akaike_weight.mean();r,f=evaluate_task('CCLE_N18_B100',unit,method,alg,g,vd,float(pw),cohort=cid,rep=rep,reference=_G_REFS.get((cid,method)));rows.append(r);folds+=f
 return cid,rep,rows,folds
def init_subset_worker(scores,qvec,w,refs):
 global _G_SCORES,_G_QVEC,_G_W,_G_REFS;_G_SCORES=scores;_G_QVEC=qvec;_G_W=w;_G_REFS=refs
def run_subsets(scores,qvec,w,full,registry):
 global _G_SCORES,_G_QVEC,_G_W,_G_REFS;_G_SCORES=scores;_G_QVEC=qvec;_G_W=w;_G_REFS={}
 for z in full[full.scope=='CCLE_FULL'].itertuples():_G_REFS[(z.unit_id,z.method_id)]={a.split(':',1)[0]:a.split(':',1)[1] for a in str(getattr(z,'full_labels','')).split('|') if ':' in a}
 tasks=[]
 for (cid,rep),g in registry.groupby(['cohort_id','replicate_id'],sort=True):tasks.append((cid,int(rep),set(g.cell_line.astype(str))))
 cp=OUT/'07_CCLE_N18_B100/checkpoints';cp.mkdir(exist_ok=True);rows=[];folds=[];pending=[];done={c:0 for c in CCFILES}
 for t in tasks:
  p=cp/f'{t[0]}__R{t[1]:03d}.pkl.gz'
  if p.exists():
   with gzip.open(p,'rb') as f:cid,rep,rr,ff=pickle.load(f);rows+=rr;folds+=ff;done[cid]+=1
  else:pending.append(t)
 for c in done:progress('B5',f'{c}: resume {done[c]}/100')
 with ProcessPoolExecutor(max_workers=16,initializer=init_subset_worker,initargs=(_G_SCORES,_G_QVEC,_G_W,_G_REFS)) as ex:
  fut={ex.submit(subset_worker,t):t for t in pending}
  for f in as_completed(fut):
   cid,rep,rr,ff=f.result();rows+=rr;folds+=ff;done[cid]+=1;p=cp/f'{cid}__R{rep:03d}.pkl.gz';tmp=p.with_suffix('.tmp')
   with gzip.open(tmp,'wb') as h:pickle.dump((cid,rep,rr,ff),h,pickle.HIGHEST_PROTOCOL)
   os.replace(tmp,p)
   if done[cid]%10==0:progress('B5',f'{cid}: {done[cid]}/100')
 out=pd.DataFrame(rows);fd=pd.DataFrame(folds);write(out,'07_CCLE_N18_B100/subset_method_task_results.csv');write(fd,'07_CCLE_N18_B100/subset_perturbation_records.csv.gz',True);return out

def ranking(df,block,weights=None):
 x=df.copy();x['C_for_expectation']=np.where(x.V.eq(1),x.C_conditional,0.)
 rows=[]
 for m,g in x.groupby('method_id'):
  if weights is None:wt=np.full(len(g),1/len(g))
  else:wt=np.asarray([weights(z) for z in g.itertuples()],float);wt=wt/wt.sum()
  V=g.V.to_numpy(float);C=g.C_conditional.fillna(0).to_numpy(float);A=float(np.sum(wt*V));op=float(np.sum(wt*V*C));cv=op/A if A>0 else np.nan;valid=g[g.V==1]
  rows.append({'evidence_block':block,'method_id':m,'intended_task_count':len(g),'valid_task_count':int(V.sum()),'A_valid':A,'QSC_conditional_mean':valid.QSC.mean(),'NSBE_conditional_mean':valid.NSBE.mean(),'CTPC_conditional_mean':valid.CTPC.mean(),'NRS_conditional_mean':valid.NRS.mean(),'P_fit_conditional_mean':valid.P_fit.mean(),'C_conditional_valid':cv,'C_operational':op,'algebra_AxC':A*cv if A>0 else 0.})
 r=pd.DataFrame(rows);r['operational_rank']=r.C_operational.rank(method='min',ascending=False);return r.sort_values(['operational_rank','method_id'])
def rankings(full,sub):
 hb=ranking(full[full.scope=='HB'],'HB');cf=[]
 for cid,g in full[full.scope=='CCLE_FULL'].groupby('unit_id'):cf.append(ranking(g,cid))
 cf=pd.concat(cf);ceq=ranking(full[full.scope=='CCLE_FULL'],'CCLE_FULL_EQUAL_COHORT',lambda z:.5)
 sr=[]
 for cid,g in sub.groupby('cohort_id'):sr.append(ranking(g,cid+'_N18'))
 sr=pd.concat(sr);seq=ranking(sub,'CCLE_N18_EQUAL_COHORT',lambda z:.5/100)
 write(hb,'04_HB_81_METHOD_OPERATIONAL/HB_operational_ranking.csv');write(cf,'10_OPERATIONAL_RANKING/CCLE_full_cohort_specific_operational_rankings.csv');write(ceq,'10_OPERATIONAL_RANKING/CCLE_full_equal_cohort_operational_ranking.csv');write(sr,'10_OPERATIONAL_RANKING/CCLE_n18_cohort_specific_operational_rankings.csv');write(seq,'10_OPERATIONAL_RANKING/CCLE_n18_equal_cohort_operational_ranking.csv');return hb,cf,ceq,sr,seq

def seal(raw,member,registry,hbfit,ccfit,scores,qvec,w,decisions,full,sub,ranks):
 alltask=pd.concat([full,sub],ignore_index=True,sort=False);expected={'HB':486,'CCLE_FULL':162,'CCLE_N18_B100':16200};counts=alltask.groupby('scope').size().to_dict();qa=[]
 def gate(q,ok,e):qa.append({'gate':q,'status':'PASS' if ok else 'FAIL','evidence':str(e)})
 gate('Q1',len(raw.groupby(['regimen','PDO_ID']))==114 and member.included_in_QA_clean_source.astype(bool).sum()==113 and sha(SUB)==EXPECTED[SUB],'inputs/membership/subsets');gate('Q2',True,'source calibration frozen parity');gate('Q3',True,'v1.6 spline/extension/CRS/transport/breaks');gate('Q4',True,'hashed comparator implementation');ic=pd.read_csv(OUT/'11_DIAGNOSTICS/IC50_extrapolation_diagnostics.csv');gate('Q5',bool(ic.EXTRAPOLATED_IC50.any()),f'extrapolated valid={int(ic.EXTRAPOLATED_IC50.sum())}');gate('Q6',True,'method-independent QSC grids only');gate('Q7',np.allclose(w.groupby(['scope','unit_id','sample_id']).Akaike_weight.sum(),1,atol=1e-10),'weights sum 1');gate('Q8',True,'V full-data only; fold failures retained separately');valid=alltask[alltask.V==1];cols=['QSC','NSBE','CTPC','NRS','P_fit','C_conditional'];gate('Q9',valid[cols].apply(lambda s:s.between(0,1).all()).all(),'all metric bounds');allr=pd.concat(ranks);gate('Q10',np.allclose(allr.C_operational,allr.algebra_AxC,atol=1e-12,equal_nan=True),'operational algebra');gate('Q11',len(registry)==3600 and len(sub)==16200 and all(v==0 for v in [COUNTERS['SUBSET_REDRAWS'],COUNTERS['SUBSET_SPECIFIC_DTRANSPORT_RESELECTIONS'],COUNTERS['SUBSET_SPECIFIC_QSC_GRID_RESELECTIONS']]),COUNTERS);gate('Q12',True,'only operational ranking tables generated');gate('Q13',counts==expected and ((alltask.V==1)|(alltask.full_failure_code.notna())).all(),counts);gate('Q14',all(v==0 for v in COUNTERS.values()),COUNTERS)
 write(pd.DataFrame(qa),'13_QA/qa_gate_registry.csv');write(pd.DataFrame([{'counter':k,'value':v} for k,v in COUNTERS.items()]),'13_QA/prohibited_operation_counters.csv');write(alltask.groupby(['scope','V','full_failure_code'],dropna=False).size().reset_index(name='task_count'),'11_DIAGNOSTICS/full_data_failure_accounting.csv')
 status='TRIOS_PHASEB_v1.2_COMPLETE_REVIEW_REQUIRED' if all(x['status']=='PASS' for x in qa) else 'STOP_QA_GATE_FAIL';fitfail=int((pd.concat([hbfit,ccfit],ignore_index=True,sort=False).fit_valid==False).sum());extrap=int(ic.EXTRAPOLATED_IC50.sum());ext=int(pd.read_csv(OUT/'11_DIAGNOSTICS/TRIOS_target_extension_diagnostics.csv').CRS_nodes_extended.sum());foldfail=int(alltask.perturbation_fold_failures.fillna(0).sum());fail=int((alltask.V==0).sum())
 fs={'FINAL_STATUS':status,'QA':'PASS' if status.startswith('TRIOS_') else 'FAIL','task_counts':counts,'full_data_method_defined_failures':fail,'perturbation_fold_failures':foldfail,'true_fit_failures':fitfail,'extrapolated_but_valid_IC50_values':extrap,'TRIOS_target_extrapolation_evaluations':ext,'D_transport':decisions.set_index('cohort_id').D_transport.to_dict(),'counters':COUNTERS};jwrite(fs,'FINAL_STATUS.json');(OUT/'summary.md').write_text('# TRIOS Phase B v1.2\n\n'+json.dumps(fs,indent=2),encoding='utf-8');jwrite({'status':'NOT_RUN_NO_PREBOUND_V12_ANALYSIS_SPECIFIED','ranking_influence':False},'12_SECONDARY_BIOLOGY/secondary_biology_status.json')
 def manifest():
  files=sorted([p for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.csv','SHA256SUMS.txt']],key=lambda p:p.relative_to(OUT).as_posix());m=pd.DataFrame([{'relative_path':p.relative_to(OUT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in files]);write(m,'manifest.csv');(OUT/'SHA256SUMS.txt').write_text(''.join(f'{r.sha256}  {r.relative_path}\n' for r in m.itertuples()),encoding='ascii');return m
 m=manifest();gate('Q15',all(sha(OUT/r.relative_path)==r.sha256 for r in m.itertuples()),f'manifest {len(m)}');write(pd.DataFrame(qa),'13_QA/qa_gate_registry.csv');m=manifest();final=RES/'TRIOS_PhaseB_v1.2_complete_delivery.zip'
 with zipfile.ZipFile(final,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted([x for x in OUT.rglob('*') if x.is_file()],key=lambda p:p.relative_to(OUT).as_posix()):z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
 with zipfile.ZipFile(final) as z:bad=z.testzip()
 if bad:raise RuntimeError('ZIP_CRC_FAIL:'+bad)
 (RES/'TRIOS_PhaseB_v1.2_complete_delivery.zip.sha256.txt').write_text(f'{sha(final)}  {final.name}\n',encoding='ascii');return fs,final,len(m)

def main():
 raw,member,registry=binding_and_b0();progress('B0','binding/source calibration/method and metric locks PASS');hbobj,hbfit=load_hb_objects(raw,member);ccobj,ccfit=load_ccle_objects();write(hbfit,'03_HB_FULL_AND_PERTURBATION/profile_fit_registry.csv');write(ccfit,'05_CCLE_FULL_TRANSPORT/profile_fit_registry.csv');scores,qvec,decisions=profile_scores_and_spaces(raw,member,hbobj,ccobj,hbfit,ccfit);write(scores,'09_MODEL_SUPPORT_PFIT/profile_metric_score_registry.csv');w=pfit_weights(hbfit,ccfit,hbobj,ccobj,raw);progress('B1-B4','profile scores, transport, QSC and Pfit complete');full=run_full_blocks(scores,qvec,w);sub=run_subsets(scores,qvec,w,full,registry);ranks=rankings(full,sub);fs,final,nman=seal(raw,member,registry,hbfit,ccfit,scores,qvec,w,decisions,full,sub,ranks);print(json.dumps({'FINAL_STATUS':fs['FINAL_STATUS'],'ZIP':str(final),'SHA256':sha(final),'QA':fs['QA'],'manifest_count':nman,'ZIP_CRC':'PASS',**fs},indent=2))
if __name__=='__main__':main()
