"""New-directory adapter. Frozen producers are imported; no parent main is called."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import hashlib,json,gzip,pickle,time,importlib.util,math
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(r'SOURCE_PROJECT')
OUT=Path(__file__).resolve().parents[1]
AUDIT=R/'deliveries/E4_EXTENSION_PHASE0_SOURCE_LOCK_20260929_v1'
CURRENT=R/'04_RESULTS/TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_20260912_195937/05_PHASE_C/E4'
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
cur=loadmod('extension_current_e4',R/'work/trios_crs_logspace_amendment_v10/run_phase_c_e4.py')
src=cur.s1mod;pf=src.pf;s1=src.s1;e1=src.e1
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def js(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n',encoding='utf8');os.replace(tmp,p)
def load(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('REFUSE_CHECKPOINT_OVERWRITE:'+str(p))
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with gzip.open(tmp,'wb',compresslevel=3) as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    q=load(tmp);assert q['key']==obj['key'] and q['complete']
    os.replace(tmp,p);return sha(p)
def geometry(D):
    g=cur.pre.crs_geometry(1.,float(D),7)
    return dict(D_transport=float(D),K=7,nodes=g['nodes'].tolist(),beta=g['beta'].tolist(),geometry_sha256=array_sha(np.r_[g['nodes'],g['beta']]))
GRIDS={f'J{j}':np.r_[0.,[150**(k/(j-1)) for k in range(j)]] for j in (3,4,6)}
GRIDS['H250']=src.e4.CONTROL.copy()
CACHES={a:src.e4.assay_cache(d,a) for a,d in GRIDS.items()}
for a,c in CACHES.items():c['M']=len(GRIDS[a])
def state_from_model(m):
    return dict(intercept=m.intercept_,differences=m.differences_,coefficients=m.coefficients_,edge=m.edge_knots_,lambda_=m.selected_lambda,EDF=m.effective_degrees_of_freedom,GCV=m.GCV,SSE=m.replicate_level_SSE,solver_status=m.solver_status,solver_iterations=m.solver_iterations,**{'lambda':m.selected_lambda})
def arrays(fits,cache):
    A=np.empty((len(fits),6));T=A.copy();C=A.copy();bad=False
    for i,f in enumerate(fits):A[i],T[i],C[i],_,b=s1.derived_arrays(f,cache);bad|=b
    return A,T,C,bad
def raw_arm(raw,arm):
    Y=raw['Y_by_assay']['H150'];n=len(raw['ids'])
    if arm in ('J3','J4'):
        inds={'J3':[0,1,4,7],'J4':[0,1,3,5,7]}[arm]
        y=Y[:,inds,:].copy();assert np.array_equal(GRIDS[arm],src.GRIDS['H150'][inds])
        return y,dict(source_indices=inds,new_replicates=0,raw_sha256=array_sha(y))
    assert arm=='J6';m=raw['meta'];block=e1.common_block(m);spec=e1.LOCK['families'][m['family']]
    _,latent=e1.latent_builder(m['family'],spec['center'],spec['deltas'][m['separation']],block)
    y=np.empty((n,7,3));y[:,[0,1,6],:]=Y[:,[0,1,7],:];mus=[]
    for i in range(n):
        key=src.paired_key(m,i+1);assert key==raw['paired_keys'][i]
        # Verify the latent function against all stored H150 means, without creating new subjects.
        assert np.array_equal(np.asarray(latent(i,src.GRIDS['H150'])),raw['mu_by_assay']['H150'][i]),'LATENT_PARITY'
        mu=np.asarray(latent(i,GRIDS[arm][2:6]),float);mus.append(mu)
        for j,d in enumerate(GRIDS[arm][2:6],2):
            for rep in range(3):y[i,j,rep]=mu[j-2]+src.normal_noise(key,format(float(d),'.17g'),rep+1)
    assert np.array_equal(y[:,[0,1,6],:],Y[:,[0,1,7],:])
    return y,dict(source_indices=[0,1,7],new_replicates=n*12,raw_sha256=array_sha(y),new_latent_means=np.asarray(mus))
def downstream(raw,arm,fits,models=None):
    m=raw['meta'];ids=raw['ids'];n=len(ids);truth=np.asarray(raw['truth'],int)
    task={k:m[k] for k in ('cohort_id','n','pattern_id','separation','base_block_id','family','n_L','n_I','n_H')}
    task.update(arm=arm,factor_weight=src.factor_weight(m),valid=False,failure_category='',K=7)
    out=dict(task=task,transport=[],loo=[],scores=[],geometries={})
    if any(f is None for f in fits):task['failure_category']='SPLINE_FAILURE';return out
    A,T,_,bad=arrays(fits,CACHES[arm])
    if bad:task['failure_category']='SPLINE_BOUNDARY_SLOPE_FAILURE';return out
    j,trans=s1.transport_index(A,T);out['transport']=trans
    if j is None:task['failure_category']='TRANSPORT_NO_PASS';task['transport_failure']=True;return out
    states=[dict(sample_id=i,state=f) for i,f in zip(ids,fits)];cache={}
    def score(D):
        D=float(D)
        if D not in cache:
            cache[D]=cur.score_models(models,D) if models is not None else cur.score_states(states,D)
            assert np.isfinite(cache[D]).all(),'NONFINITE_SCORES'
            out['geometries'][str(D)]=geometry(D)
        return cache[D]
    D=float(pf.TCANDS[j]);x=score(D);z=cur.ab(x,ids);task['D_transport']=D
    if z is None:task['failure_category']='AB_NO_ADMISSIBLE_PARTITION';return out
    lab,t1,t2=z;folds=[];Rfull=float(np.ptp(x));assert Rfull>0
    for h in range(n):
        keep=np.arange(n)!=h;jh,reg=s1.transport_index(A,T,keep)
        r=dict(fold=h+1,omitted_sample_id=ids[h],success=False,failure_code='TRANSPORT_NO_PASS',D_transport=np.nan,K=7)
        if jh is not None:
            Dh=float(pf.TCANDS[jh]);xx=score(Dh);zz=cur.ab(xx[keep],np.asarray(ids)[keep]);r.update(D_transport=Dh,geometry_sha256=out['geometries'][str(Dh)]['geometry_sha256'],failure_code='AB_NO_ADMISSIBLE_PARTITION')
            if zz is not None:
                ll,a,b=zz;held=0 if xx[h]<=a else (1 if xx[h]<=b else 2)
                r.update(success=True,failure_code='',tau1=float(a),tau2=float(b),omitted_score=float(xx[h]),omitted_label=held,retained_scores=xx[keep],retained_labels=ll,omitted_match=float(held==lab[h]),retained_match=float(np.mean(ll==lab[keep])),d_i_squared=.5*(((a-t1)/Rfull)**2+((b-t2)/Rfull)**2))
        folds.append(r)
    # Call the frozen current evaluator for the actual metrics; it consumes our native fold domains.
    t,sc,_=cur.eval_arm(task|{'valid':True},folds,ids,truth,score)
    assert t['valid'];assert t['A_LOO']==sum(r['success'] for r in folds)/n
    for r in sc:r.update(arm=arm,cohort_id=m['cohort_id'])
    out.update(task=t,loo=folds,scores=sc)
    return out
def parity_B(raw,old,fits):
    errs=[]
    for pid,f,o in zip(raw['ids'],fits,old['fits']):
        assert pid==o['sample_id'] and bool(o['fit_valid'])==(f is not None),'FIT_STATUS_ID_PARITY'
        if f is not None:
            assert f['lambda']==float(o['selected_lambda']),'LAMBDA_PARITY'
            e={k:abs(float(f[k])-float(o[k])) for k in ('SSE','GCV','EDF')}
            assert max(e.values())<=1e-9,('FIT_DIAGNOSTIC_PARITY',pid,e)
            errs.append(dict(sample_id=pid,**e))
    if any(f is None for f in fits):raise RuntimeError('H250_HISTORICAL_INVALID_FIT_REPRODUCTION_UNRESOLVED')
    A,T,C,bad=arrays(fits,CACHES['H250']);assert not bad,'BOUNDARY_PARITY'
    j,reg=s1.transport_index(A,T);D=None if j is None else float(pf.TCANDS[j]);od=old['task'].get('D_transport')
    assert (D is None and (od is None or not np.isfinite(od))) or D==od,('TRANSPORT_PARITY',D,od)
    for x,o in zip(reg,old['transport']):
        assert bool(x[4])==bool(o['candidate_pass']) and x[5]==o['stage_evaluable_n'],'TRANSPORT_CANDIDATE_PARITY'
        for v,k in zip(x[1:4],('E_stage','A_e','T_e')):
            assert (np.isnan(v) and np.isnan(o[k])) or abs(v-o[k])<=1e-9,('TRANSPORT_VALUE_PARITY',k)
    if old['task']['valid']:
        assert len(old['loo'])==len(raw['ids'])
        z=s1.ab(C[:,j],raw['ids']);assert z is not None
        for i,o in enumerate(old['scores']):
            assert o['sample_id']==raw['ids'][i] and int(o['label'])==int(z[0][i])
            assert abs(C[i,j]-o['CRS'])<=1e-9,'HISTORICAL_GEOMETRY_DIAGNOSTIC'
        for h,o in enumerate(old['loo']):
            keep=np.arange(len(raw['ids']))!=h;jh,_=s1.transport_index(A,T,keep);dh=np.nan if jh is None else float(pf.TCANDS[jh]);od=o.get('D_transport',np.nan)
            assert o['omitted_sample_id']==raw['ids'][h]
            assert (np.isnan(dh) and np.isnan(od)) or dh==od,'FOLD_DOMAIN_PARITY'
    else:assert j is None and old['task']['failure_category']=='TRANSPORT_NO_PASS','FULL_FAILURE_PARITY'
    return dict(status='PASS',profiles=errs,transport='PASS',historical_geometry='DIAGNOSTIC_ONLY')
def checkpoint_job(arg):
    lane,arm,sh,index,config=arg;start=time.perf_counter()
    rawpath=(src.RAW if lane=='A' else src.V2_WORK/'raw_bank')/f'raw_shard_{sh:02d}.pkl.gz'
    raw=load(rawpath)['cohorts'][index];cid=raw['meta']['cohort_id'];key=[lane,arm,cid,sha(rawpath),config]
    dest=OUT/f'LANE_{lane}/RAW_FITS/{arm}/{cid}.pkl.gz'
    if dest.exists():
        obj=load(dest);assert obj['key']==key and obj['complete'];return dict(lane=lane,arm=arm,cohort_id=cid,profiles=len(raw['ids']),reused=True,seconds=0,path=str(dest))
    claim=dest.with_suffix('.claim');claim.parent.mkdir(parents=True,exist_ok=True)
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),key=key,time=time.time()),f)
    if lane=='A':
        y,provenance=raw_arm(raw,arm);fits=[pf.cached_fit(v.reshape(-1),CACHES[arm]) for v in y]
        result=downstream(raw,arm,fits);parity=None
    else:
        old=load(src.V2_WORK/'checkpoints'/f'fit_shard_{sh:02d}.pkl.gz')['cohorts'][index]
        assert old['cohort_id']==cid;old=old['REGENERATED_E1_GRID'];y=raw['Y_control'];fits=[]
        for v,f in zip(y,old['fits']):
            if not f['fit_valid']:raise RuntimeError('H250_OLD_INVALID_STATE_REQUIRES_ADJUDICATION')
            model=pf.fit_constrained_pspline(CACHES[arm]['z'],v.reshape(-1),float(f['selected_lambda']),CACHES[arm]['edge'],n_splines=12,spline_order=3);fits.append(state_from_model(model))
        parity=parity_B(raw,old,fits);provenance=dict(source='REGENERATED_E1_GRID',new_replicates=0,raw_sha256=array_sha(y));result=None
    obj=dict(complete=True,key=key,meta=raw['meta'],ids=raw['ids'],truth=raw['truth'],paired_keys=raw['paired_keys'],raw=y,raw_provenance=provenance,fits=fits,result=result,parity=parity,seconds=time.perf_counter()-start)
    digest=atomic(dest,obj);claim.unlink()
    return dict(lane=lane,arm=arm,cohort_id=cid,profiles=len(raw['ids']),reused=False,seconds=obj['seconds'],path=str(dest),sha256=digest)
