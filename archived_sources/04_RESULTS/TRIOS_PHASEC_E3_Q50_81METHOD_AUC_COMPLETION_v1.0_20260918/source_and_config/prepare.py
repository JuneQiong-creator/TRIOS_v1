"""Bind frozen observations/curves and certified representations. No fitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import ast,gzip,hashlib,importlib.util,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from bind import ROOT,E3,P,OUT,sha
E1=ROOT/'04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0'
REC=E1.parent/'TRIOS_PHASEC_SIM_E1_v1.0_ARCHIVE_v1.1_FITTED_CURVE_RECOVERED_FROZEN'
ORIG=ROOT/'work/trios_phasec_sim_e1_v10/checkpoints_authoritative'
WORK=Path(__file__).parent
def load_recovery():
 spec=importlib.util.spec_from_file_location('frozen_recovery',ROOT/'work/trios_phasec_sim_e1_curve_recovery_v11/run_recovery_v11.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def forbidden(*a,**kw):raise RuntimeError('FORBIDDEN_SPLINE_REFIT')
 m.fit_constrained_pspline=forbidden
 return m
def read(p):return pd.read_csv(p,float_precision='round_trip')
def main():
 assert json.loads((OUT/'00_BINDING/resumed_scientific_parent_v1/BINDING_STATUS.json').read_text())['PARENT_BINDING_STATUS']=='PASS'
 dest=WORK/'inputs';dest.mkdir(exist_ok=True)
 coh=read(P/'binding/cohort_registry.csv');ids=set(coh.cohort_id)
 man=read(REC/'manifest_v1.1.csv').set_index('path')
 state={};bound=[]
 for p in sorted((REC/'05_SPLINE_FITS/recovered_curve_representation').glob('*.csv.gz')):
  digest=sha(p);assert digest==str(man.loc[p.relative_to(REC).as_posix(),'sha256']).lower()
  d=read(p);d=d[d.cohort_id.isin(ids)]
  for r in d.to_dict('records'):
   for k in ('differences','coefficients','edge_knots'):r[k]=ast.literal_eval(r[k])
   state[r['sample_id']]=r
  bound.append(dict(path=str(p),sha256=digest))
 assert len(state)==53100
 fits=read(P/'06_MERGED/authoritative_merged_fitting_outputs.csv.gz');fg={cid:g.set_index(['sample_id','model']) for cid,g in fits.groupby('cohort_id')}
 profiles=read(P/'binding/profile_registry.csv.gz').set_index('sample_id')
 rec=load_recovery();parity=[];weightrows=[];inputrows=[]
 for j,meta in enumerate(coh.to_dict('records'),1):
  cid=meta['cohort_id'];path=ORIG/(cid+'.pkl');raw=pickle.load(path.open('rb'))
  sample=sorted(x['sample_id'] for x in raw['latent']);truth={x['sample_id']:x['truth'] for x in raw['latent']}
  assert len(sample)==meta['n'] and set(sample)==set(fg[cid].index.get_level_values(0))
  for sid in sample:
   rr=sorted((x for x in raw['raw'] if x['sample_id']==sid),key=lambda x:(x['dose_uM'],x['technical_replicate']))
   data=[x['dose_uM'] for x in rr]+[x['observed_response'] for x in rr]
   assert hashlib.sha256(np.asarray(data,dtype='<f8').tobytes()).hexdigest()==profiles.loc[sid,'raw_hash'].lower()
   assert truth[sid]==profiles.loc[sid,'truth']
  preds=[rec.pred_from_record(state[sid]) for sid in sample]
  endpoint,candidates=rec.transport(preds)
  assert endpoint==raw['estimated'],('TRANSPORT_PARITY',cid)
  parent=next((x for x in raw['scenarios'] if x['scenario']=='E1C_C_COMPLETE_ESTIMATED'),None)
  labels={x['sample_id']:x['predicted'] for x in raw['labels'] if x['scenario']=='E1C_C_COMPLETE_ESTIMATED'}
  scores=rec.crs(preds,endpoint) if endpoint is not None else np.full(len(sample),np.nan)
  if parent is not None:
   rebuilt=rec.scenario('E1C_C_COMPLETE_ESTIMATED',scores,sample,np.array([truth[s] for s in sample]))
   ok,field=rec.compare_scenario(rebuilt,parent,labels);assert ok,(cid,field)
  else:assert endpoint is None
  aics=[]
  for sid in sample:
   sr=next(x for x in raw['fits'] if x['sample_id']==sid);N=24;K=float(sr['EDF'])+1;sse=float(sr['SSE'])
   sa=N*np.log(sse/N)+2*K+2*K*(K+1)/(N-K-1) if N>K+1 else np.inf
   vals=[sa]+[float(fg[cid].loc[(sid,m),'AICc']) for m in ('RAW_EMAX','RAW_HILL','RAW_LOGISTIC','RAW_WEIBULL')]
   vals=np.array(vals);vv=np.where(np.isfinite(vals),vals,np.inf);assert np.isfinite(vv).any()
   w=np.exp(-.5*(vv-vv.min()));w/=w.sum();assert abs(w.sum()-1)<=1e-12;aics.append(w)
   for model,a,ww in zip(('SPLINE','EMAX','HILL','LOGISTIC','WEIBULL'),vals,w):weightrows.append(dict(cohort_id=cid,sample_id=sid,model=model,AICc=a,weight=ww))
  payload=dict(meta=meta,ids=sample,truth=[truth[s] for s in sample],curve_states=[state[s] for s in sample],parametric=fg[cid].reset_index().to_dict('records'),P_fit=dict(zip(('SPLINE','EMAX','HILL','LOGISTIC','WEIBULL'),np.mean(aics,axis=0))),TRIOS_parent=parent,TRIOS_labels=labels,TRIOS_endpoint=endpoint,TRIOS_scores=scores)
  target=dest/(cid+'.pkl.gz')
  with gzip.open(target,'wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
  inputrows.append(dict(cohort_id=cid,path=str(target),sha256=sha(target),raw_checkpoint_sha256=sha(path)))
  parity.append(dict(cohort_id=cid,profiles=len(sample),raw_identity='PASS',transport_parity='PASS',Level_C_parity='PASS',valid=bool(parent and parent['valid'])))
  if j%100==0:print(f'[{time.strftime("%H:%M:%S")}] Frozen bank/TRIOS parity {j}/2550',flush=True)
 (OUT/'01_TRIOS_PARITY').mkdir(exist_ok=True);(OUT/'02_MODEL_SUPPORT').mkdir(exist_ok=True)
 pd.DataFrame(parity).to_csv(OUT/'01_TRIOS_PARITY/full_bank_parity.csv',index=False)
 pd.DataFrame(weightrows).to_csv(OUT/'02_MODEL_SUPPORT/profile_AICc_weights.csv.gz',index=False,float_format='%.17g')
 pd.DataFrame(inputrows).to_csv(WORK/'prepared_inputs.csv',index=False)
 pd.DataFrame(bound).to_csv(OUT/'00_BINDING/curve_state_binding.csv',index=False)
 (WORK/'PREPARED.json').write_text(json.dumps(dict(status='PASS',cohorts=2550,profiles=53100,TRIOS_parity=2550,parametric_refits=0,spline_refits=0)),encoding='utf-8')
 print('PREPARED PASS: 2550 cohorts; 53100 profiles; no refits',flush=True)
if __name__=='__main__':main()
