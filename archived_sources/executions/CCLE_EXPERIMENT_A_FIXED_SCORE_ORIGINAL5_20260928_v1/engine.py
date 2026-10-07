"""Experiment A only: frozen vectors and native classifier-only LOO.
No upstream producer, fit, domain selector, CRS reconstruction or integration is executed.
"""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,hashlib,gzip,math,time,functools
import durable_runtime as rt
A=Path(__file__).resolve().parent;ROOT=A.parents[1]
F=A.parent/'CCLE_OPTION_C_MIN_INCREMENTAL_FULL_20260928_v1';O=A.parent/'CCLE_OPTION_C_COMPLETE_20260928_v1'
sp=importlib.util.spec_from_file_location('approved_components_library',F/'execute_full.py');e=importlib.util.module_from_spec(sp);sys.modules[sp.name]=e;sp.loader.exec_module(e)
e.O=A;e.STOP=A/'STOP_REQUESTED';np=e.np;CODE=rt.sha(__file__)
CONTRACT='CCLE_EXPERIMENT_A_FIXED_SCORE_ORIGINAL5'
METHODS=['TRIOS']+[f'{m}|{r}|{a}' for m in e.MODELS for r in e.METRICS for a in e.ALGS]
COMP=['S_BW','S_tau','S_match','P_fit','I_SF']
def forbid(*a,**k):raise e.IntegrityStop('EXPERIMENT_A_UPSTREAM_COMPUTATION_FORBIDDEN')
e.select_domain=forbid;e.quad=forbid;e.brentq=forbid;e.frozen=forbid;e.spline.fit_constrained_pspline=forbid
def xhash(x,ids):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()+b'|'+('|'.join(ids)).encode()).hexdigest()
def key(method):return hashlib.sha256(method.encode()).hexdigest()[:16]
def num(x):return float(x) if x not in ('',None) else math.nan
@functools.lru_cache(None)
def binding(cid):return rt.read(A/'00_BINDING'/f'{cid}.json')
def bound_read(path,expected):
    raw=Path(path).read_bytes();e.require(hashlib.sha256(raw).hexdigest()==expected,'SOURCE_SHA_MISMATCH:'+str(path));return json.loads(raw.decode('utf-8-sig'))
def save(p,r):rt.atomic_bytes(p,gzip.compress((json.dumps(e.clean(r),allow_nan=False,separators=(',',':'))+'\n').encode(),mtime=0),immutable=True)
def read_checkpoint(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def location(ctx):return A/('02_FULL' if ctx['scope']=='FULL' else '03_B100')/ctx['dataset_id']/('FULL' if ctx['scope']=='FULL' else f"R{ctx['rep']:03d}")

def evaluate_context(ctx):
    start=time.time();e.stopped();cid=ctx['dataset_id'];b=binding(cid);ids=ctx['ids'];n=len(ids);dest=location(ctx);dest.mkdir(parents=True,exist_ok=True)
    with rt.Lease(rt.RUNTIME/'tasks'/(hashlib.sha256(ctx['unit'].encode()).hexdigest()[:20]+'.lock'),ctx['unit']):
        if (dest/'COMPLETE.json').exists():
            out=rt.read(dest/'COMPLETE.json');e.require(out['code_sha256']==CODE,'CONTEXT_CODE');return out
        counts=dict(completed=0,reused_conventional_tasks=0,reused_fold_slots=0,computed_classifiers=0,computed_TRIOS_folds=0,native_TRIOS_folds_reused=0)
        for method in METHODS:
            e.stopped();path=dest/(key(method)+'.json.gz')
            if path.exists():
                saved=read_checkpoint(path);e.require(saved['code_sha256']==CODE and saved['result']['method_id']==method,'CHECKPOINT_IDENTITY')
                for name,value in saved['counts'].items():counts[name]+=value
                continue
            oldpath=Path(ctx['checkpoint_dir'])/(key(method)+'.task.json');old=bound_read(oldpath,ctx['checkpoint_sha256'][method]);r=old['result'];trios=method=='TRIOS';model,metric,alg=('SPLINE','CRS','AB') if trios else method.split('|')
            x=np.array([b['scores'][pid]['TRIOS' if trios else model+'|'+metric] for pid in ids],dtype=float)
            D=b['D2_domain'] if trios else 8.;hx=xhash(x,ids);pf=float(np.mean([b['weights'][pid][model] for pid in ids]))
            seed=f"TRIOS_PHASEB_V12|{'CCLE_FULL' if ctx['scope']=='FULL' else 'CCLE_N18_B100'}|{ctx['unit']}|{method}"
            counter=dict(completed=1,reused_conventional_tasks=0,reused_fold_slots=0,computed_classifiers=0,computed_TRIOS_folds=0,native_TRIOS_folds_reused=0)
            full_diag={};failure=''
            if trios:
                rr=e.classify(x,np.array(ids),alg,seed+'|FULL');counter['computed_classifiers']+=1;V=bool(rr['valid']);full_diag=e.clean(rr)
                t1=rr.get('tau1') if V else None;t2=rr.get('tau2') if V else None;labs=np.array(rr['labels']) if V else None;failure=rr.get('failure_code','CLASSIFIER_FAILURE') if not V else ''
                if ctx['scope']=='FULL':e.require(V==bool(r['V']) and (not V or (t1==r['tau1'] and t2==r['tau2'] and labs.tolist()==r['full_labels'])),'FULL_D2_AB_SOURCE_PARITY')
                if V:e.require(min(np.sum(labs==g) for g in 'LIH')>=max(3,math.ceil(.15*n)),'FULL_AB_MIN')
            else:
                e.require(r['method_id']==method and r['task_id']==ctx['unit'] and r['n']==n and r['D']==8.,'CONVENTIONAL_KEY_DOMAIN')
                if 'full_scores' in r:e.require(np.array_equal(x,np.array(r['full_scores'],float),equal_nan=True),'CONVENTIONAL_SCORE_PARITY')
                V=bool(r['V']);t1=r['tau1'];t2=r['tau2'];labs=np.array(r['full_labels']) if V else None;failure=r.get('full_failure_code') or '';counter['reused_conventional_tasks']=1
            if V:e.require(np.isfinite(x).all() and np.ptp(x)>0 and np.array_equal(e.project(x,t1,t2,trios),labs),'FULL_PROJECTION_OR_FINITE')
            folds=[]
            if not trios:e.require(len(old['folds'])==n and [q['heldout_profile'] for q in old['folds']]==ids,'SOURCE_FOLD_KEYS')
            for h,pid in enumerate(ids):
                e.stopped();keep=np.arange(n)!=h
                q=dict(dataset_id=cid,scope=ctx['scope'],replicate_id=ctx['rep'],task_id=ctx['unit'],method_id=method,heldout_profile=pid,fold_index=h+1,n_intended=n,attempted=V,success=False,status='NOT_ATTEMPTED',D_full=D,D_fold=D if V else None,full_score_hash=hx,fold_score_hash=hx if V else None,tau1=None,tau2=None,d_tau_squared=None,omitted_match=None,retained_match_fraction=None,heldout_label=None,retained_labels=None,failure_code=None,source_path=None,source_kind='NOT_ATTEMPTED_FULL_INVALID')
                if not trios:
                    oq=old['folds'][h];e.require(bool(oq['attempted'])==V and oq['method_id']==method and oq['task_id']==ctx['unit'],'SOURCE_FOLD_PROTOCOL');counter['reused_fold_slots']+=1
                    q.update(source_path=str(oldpath),source_kind='SHA_BOUND_CONVENTIONAL_FIXED_SCORE_FOLD')
                    if V:
                        e.require(oq['D_fold']==8.,'CONVENTIONAL_FOLD_DOMAIN');q.update(success=bool(oq['success']),status='SUCCESS' if oq['success'] else 'FAILED',failure_code=oq.get('failure_code'))
                        if oq['success']:
                            e.require(np.array_equal(x,np.asarray(oq['fold_scores'],float),equal_nan=True),'STORED_FIXED_FOLD_SCORE_PARITY')
                            for field in ['tau1','tau2','d_tau_squared','omitted_match','retained_match_fraction','heldout_label','retained_labels']:q[field]=oq[field]
                        else:e.require(all(oq.get(k) is None for k in ['tau1','tau2','d_tau_squared','omitted_match','retained_match_fraction']),'FAILED_FOLD_UNDEFINED')
                elif V:
                    fr=e.classify(x[keep],np.array(ids)[keep],alg,seed+'|PERTURB|'+pid);counter['computed_classifiers']+=1;counter['computed_TRIOS_folds']+=1
                    if fr['valid']:
                        pred=e.project(x,fr['tau1'],fr['tau2'],True)
                        q.update(success=True,status='SUCCESS',source_kind='NEW_FIXED_SCORE_AB_ONLY',tau1=fr['tau1'],tau2=fr['tau2'],d_tau_squared=float(.5*sum(((fr[k]-v)/np.ptp(x))**2 for k,v in [('tau1',t1),('tau2',t2)])),omitted_match=float(pred[h]==labs[h]),retained_match_fraction=float(np.mean(pred[keep]==labs[keep])),heldout_label=str(pred[h]),retained_labels=pred[keep].tolist())
                    else:q.update(status='FAILED',source_kind='NEW_FIXED_SCORE_AB_ONLY',failure_code=fr.get('failure_code','CLASSIFIER_FAILURE'))
                folds.append(q)
            comp=e.task_components(x,labs,t1,t2,pf,folds,V)
            result=dict(contract_id=CONTRACT,dataset_id=cid,scope=ctx['scope'],replicate_id=ctx['rep'],task_id=ctx['unit'],method_id=method,n=n,D=D,domain_policy='FIXED_D2' if trios else 'FIXED_NATIVE_REPRESENTATION',perturbation='FIXED_SCORE_CLASSIFIER_ONLY',full_score_hash=hx,tau1=t1,tau2=t2,full_scores=e.clean(x),full_labels=labs.tolist() if V else None,full_failure_code=failure,**comp)
            save(path,dict(code_sha256=CODE,result=result,ids=ids,folds=folds,counts=counter,full_AB_diagnostics=full_diag,source_checkpoint=str(oldpath),source_checkpoint_sha256=ctx['checkpoint_sha256'][method]))
            for name,value in counter.items():counts[name]+=value
            if trios and not V:
                rt.atomic_json(dest/'TRIOS_INVALID_REVIEW_REQUIRED.json',dict(result=result,diagnostics=full_diag,release_dataset_ranking=False),immutable=True)
        out=dict(code_sha256=CODE,dataset_id=cid,scope=ctx['scope'],rep=ctx['rep'],tasks=81,counts=counts,elapsed_seconds=time.time()-start,status='COMPLETE_PENDING_QA')
        rt.atomic_json(dest/'COMPLETE.json',out,immutable=True);return out
